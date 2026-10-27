"""Per-user long-term profiles for AstrBot.

Each sender gets a small fact list (at most MAX_FACTS entries). Facts are
extracted asynchronously after a reply is produced and injected into the
system prompt the next time that sender talks.

Commands:
    /记忆            show your own profile
    /记忆 <QQ 号>    show someone else's profile (admin only)
    /忘记            clear your own profile
"""

import asyncio
import difflib
import json
import re
import sqlite3
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

PROFILE_FILE = Path(__file__).resolve().parent / "profiles.json"
# 插件目录是 data/plugins/user_memory，往上两级是 data/
DATA_DIR = Path(__file__).resolve().parents[2]
AFFECTION_DB = DATA_DIR / "plugin_data" / "astrbot_plugin_self_learning" / "messages.db"
MAX_FACTS = 30                 # 每人最多保留多少条（超出才压缩）
SIMILARITY_THRESHOLD = 0.85    # 归一化后相似度达到这个值视为重复
COMPACT_THRESHOLD = 15         # 超过这么多条就触发压缩
COMPACT_TARGET = 10            # 压缩后保留的条数
COMPACT_AT = (4, 30)           # 每周一 04:30 全量压缩一次
# Same sender will not trigger another extraction within this window (seconds).
MIN_INTERVAL = 300           # 同一个号两次抽取的最小间隔（秒）

EXTRACT_SYSTEM = (
    "你是一个信息抽取器。只输出 JSON，不要解释。"
    '输出格式：{"facts": ["...", "..."]}，没有可记的就输出 {"facts": []}。'
)

EXTRACT_TEMPLATE = """下面是「用户」和「助手」的一轮对话，请判断其中有没有关于这个用户的**长期稳定**信息。

可以记：职业、身份（如号主、学生、后端）、长期爱好、常用工具或技术栈、明确表达过的偏好和雷点、长期在做的事。
不要记：称呼和叫法（"爸""哥"这类随时会改，由人设定，记了会跟人设打架）、一次性的情绪、当天吃了什么、临时的安排、礼貌性寒暄、助手自己说过的话。

已有档案：
{existing}

本轮对话：
用户：{user_text}
助手：{reply}

请输出合并后的**完整**档案（已有 + 新增，去重，最多 {max_facts} 条），只输出 JSON。"""

COMPACT_PROMPT = """下面是一个用户的长期档案，可能包含重复、过时或互相矛盾的条目。

请合并成一份精简档案：
- 去掉重复：同一件事的不同说法只留最清楚的一条。
- 去掉过时和一次性的内容。
- 互相矛盾时，保留列表中更靠后的那条（它更新）。
- 保留身份、职业、长期偏好、雷点、常用技术栈这类稳定信息。
- 最多输出 {target} 条。

档案：
{existing}

只输出 JSON：{{"facts": ["...", "..."]}}"""

FACTS_KEY = "facts"


class UserMemory(Star):
    """给每个群友建一份长期档案。

    自动抽取稳定信息存本地，下次对话时注入 system prompt。
    """

    def __init__(self, context: Context, config=None) -> None:
        super().__init__(context)
        self.cfg = config
        self.enable = self._cfg("enable", True, bool)
        self.inject_profile = self._cfg("inject_profile", True, bool)
        self.inject_affection = self._cfg("inject_affection", True, bool)
        self.extract_enabled = self._cfg("extract_enabled", True, bool)
        self.min_interval = self._cfg("min_interval_seconds", MIN_INTERVAL, int)
        self.min_msg_len = self._cfg("min_message_length", 4, int)
        self.max_facts = self._cfg("max_facts", MAX_FACTS, int)
        self.similarity_threshold = self._cfg(
            "similarity_threshold", SIMILARITY_THRESHOLD, float
        )
        self.compact_enabled = self._cfg("compact_enabled", True, bool)
        self.compact_threshold = self._cfg("compact_threshold", COMPACT_THRESHOLD, int)
        self.compact_target = self._cfg("compact_target", COMPACT_TARGET, int)
        self.weekly_compact = self._cfg("weekly_compact", True, bool)
        self._last_run: dict[str, float] = {}
        self._task: asyncio.Task | None = None
        self._last_compact_day = ""
        self._io_lock = asyncio.Lock()

    def _cfg(self, key: str, default, cast):
        """读插件配置项，读不到就用默认值。

        Args:
            key: 配置键名。
            default: 缺省值。
            cast: 目标类型。

        Returns:
            转换后的配置值。
        """
        if not self.cfg:
            return default
        try:
            value = self.cfg.get(key, default)
        except Exception:  # noqa: BLE001
            return default
        if value is None:
            return default
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    # ---------- storage ----------

    def _load(self) -> dict:
        if not PROFILE_FILE.exists():
            return {}
        try:
            return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 读取档案失败: {e}")
            return {}

    def _save(self, data: dict) -> None:
        try:
            tmp = PROFILE_FILE.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(PROFILE_FILE)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 写入档案失败: {e}")

    # ---------- hooks ----------

    @filter.on_llm_request()
    async def inject_profile(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """把当前说话人的档案和好感度追加进 system prompt。"""
        if not self.enable:
            return
        uid = str(event.get_sender_id() or "").strip()
        if not uid:
            return

        blocks: list[str] = []

        prof = self._load().get(uid) or {}
        facts = [f for f in prof.get(FACTS_KEY, []) if f] if self.inject_profile else []
        if facts:
            nick = event.get_sender_name() or prof.get("name") or uid
            lines = [
                "关于正在跟你说话的这个人（你以前记下来的，别当新信息问）：",
                f"- QQ {uid}，昵称 {nick}",
            ]
            lines += [f"- {f}" for f in facts]
            lines.append("自然地用这些信息就行，别复述，也别一条条念出来。")
            blocks.append("\n".join(lines))

        # 好感度：群聊按群号查，私聊按对方 QQ 号查
        gid = str(event.get_group_id() or uid)
        affection = self._read_affection(uid, gid) if self.inject_affection else None
        if affection is not None:
            blocks.append(
                f"你对这个人的好感度是 {affection}/100。"
                "按人设里的分档决定亲近程度，别把数字念出来，也别问对方好感度多少。"
            )

        if blocks:
            req.system_prompt = (req.system_prompt or "") + "\n\n" + "\n\n".join(blocks)

    def _read_affection(self, uid: str, gid: str) -> int | None:
        """按账号读好感度：取该 QQ 在所有会话里的最高值。

        self_learning 按 (群, 用户) 分开存，但人只有一个，所以这里取最大值，
        让私聊和群里看到的是同一个数字。

        Args:
            uid: 对方的 QQ 号。
            gid: 当前会话标识，仅用于兜底（按账号查不到时）。

        Returns:
            好感度数值；查不到返回 None。
        """
        if not AFFECTION_DB.exists():
            return None
        try:
            con = sqlite3.connect(f"file:{AFFECTION_DB}?mode=ro", uri=True)
            row = con.execute(
                "SELECT MAX(affection_level) FROM user_affections WHERE user_id=?",
                (uid,),
            ).fetchone()
            con.close()
            if row and row[0] is not None:
                return int(row[0])
            return None
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 读取好感度失败: {e}")
            return None

    @filter.on_llm_response()
    async def collect_facts(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        """回复完成后，后台异步抽取这个用户的稳定信息。"""
        if not self.enable or not self.extract_enabled:
            return
        uid = str(event.get_sender_id() or "").strip()
        if not uid:
            return
        now = time.time()
        if now - self._last_run.get(uid, 0) < self.min_interval:
            return
        self._last_run[uid] = now

        user_text = (event.get_message_str() or "").strip()
        reply = (getattr(resp, "completion_text", "") or "").strip()
        if len(user_text) < self.min_msg_len or not reply:
            return
        asyncio.create_task(
            self._extract(uid, event.get_sender_name() or uid, user_text, reply)
        )

    async def _extract(self, uid: str, nick: str, user_text: str, reply: str) -> None:
        """调用一次模型，把这一轮里的稳定信息合并进档案。"""
        try:
            provider = await self.context.get_using_provider_async()
            if not provider:
                return

            async with self._io_lock:
                data = self._load()
            prof = data.get(uid) or {"name": nick, "facts": []}
            existing = prof.get(FACTS_KEY, [])

            prompt = EXTRACT_TEMPLATE.format(
                existing=json.dumps(existing, ensure_ascii=False),
                user_text=user_text[:800],
                reply=reply[:800],
                max_facts=self.max_facts,
            )
            resp = await provider.text_chat(
                prompt=prompt, system_prompt=EXTRACT_SYSTEM, persist=False
            )
            new_facts = self._parse_facts(getattr(resp, "completion_text", ""))
            if new_facts is None:
                return

            merged = self._dedupe_facts(list(existing) + list(new_facts))
            merged = merged[-self.max_facts:]

            async with self._io_lock:
                data = self._load()
                data[uid] = {
                    "name": nick,
                    "facts": merged,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._save(data)

            if self.compact_enabled and len(merged) > self.compact_threshold:
                asyncio.create_task(self._compact(uid))
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 抽取失败 uid={uid}: {e}")

    # ---------- 去重 ----------

    @staticmethod
    def _normalize_fact(text: str) -> str:
        """把一条事实归一化成可比较的形式。

        去掉标点、括号注释、空白和常见前缀（「职业：」这类），便于比较。

        Args:
            text: 原始事实文本。

        Returns:
            归一化后的字符串。
        """
        text = str(text or "").strip().lower()
        # 去掉括号里的补充说明
        text = re.sub(r"[（(][^）)]*[）)]", "", text)
        # 去掉“偏好：”“技术栈：”这类前缀
        text = re.sub(r"^[\u4e00-\u9fa5a-z]{1,6}[：:]", "", text)
        # 去掉标点和空白
        text = re.sub(r"[\s,，。.、;；:：!！?？\"'“”‘’()（）\[\]【】]", "", text)
        return text

    def _dedupe_facts(self, facts: list[str]) -> list[str]:
        """按归一化相似度去重，保留较新的那条。

        Args:
            facts: 按时间从旧到新排列的事实列表。

        Returns:
            去重后的事实列表，顺序仍是从旧到新。
        """
        kept: list[tuple[str, str]] = []   # (归一化, 原文)
        for fact in facts:
            fact = str(fact).strip()
            if not fact:
                continue
            norm = self._normalize_fact(fact)
            if not norm:
                continue

            duplicate = False
            for idx, (old_norm, _) in enumerate(kept):
                if not old_norm:
                    continue
                ratio = difflib.SequenceMatcher(None, norm, old_norm).ratio()
                shorter, longer = sorted((norm, old_norm), key=len)
                contains = len(shorter) >= 6 and shorter in longer
                if ratio >= self.similarity_threshold or (contains and len(shorter) / len(longer) >= 0.6):
                    kept[idx] = (norm, fact)   # 用较新的措辞替换旧的
                    duplicate = True
                    break
            if not duplicate:
                kept.append((norm, fact))
        return [original for _, original in kept]

    @staticmethod
    def _parse_facts(text: str) -> list[str] | None:
        """Extract the facts array from a possibly fenced JSON reply."""
        if not text:
            return None
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return None
        facts = payload.get(FACTS_KEY)
        if not isinstance(facts, list):
            return None
        return [str(f) for f in facts]

    # ---------- 压缩 ----------

    async def _compact(self, uid: str) -> int:
        """把一个人的档案压缩到 COMPACT_TARGET 条以内。

        Args:
            uid: 目标 QQ 号。

        Returns:
            压缩前的事实条数；未压缩返回 0。
        """
        if not self.compact_enabled:
            return 0
        async with self._io_lock:
            data = self._load()
        prof = data.get(uid) or {}
        facts = [f for f in prof.get(FACTS_KEY, []) if f]
        if len(facts) < self.compact_threshold:
            return 0

        try:
            provider = await self.context.get_using_provider_async()
            if not provider:
                return 0
            prompt = COMPACT_PROMPT.format(
                target=self.compact_target,
                existing=json.dumps(facts, ensure_ascii=False),
            )
            resp = await provider.text_chat(
                prompt=prompt, system_prompt=EXTRACT_SYSTEM, persist=False
            )
            compacted = self._parse_facts(getattr(resp, "completion_text", ""))
            if not compacted:
                return 0
            compacted = self._dedupe_facts(compacted)[-self.max_facts:]

            async with self._io_lock:
                data = self._load()
                entry = data.get(uid) or {"name": ""}
                entry[FACTS_KEY] = compacted
                entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                data[uid] = entry
                self._save(data)
            logger.info(f"[user_memory] 压缩 {uid}: {len(facts)} -> {len(compacted)} 条")
            return len(facts)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 压缩失败 {uid}: {e}")
            return 0

    @filter.on_astrbot_loaded()
    async def start_compact_loop(self) -> None:
        """每周一凌晨压缩一遍所有档案。"""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._compact_loop())
            logger.info("[user_memory] 档案压缩定时器已启动")

    async def _compact_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                now = time.localtime()
                if not self.weekly_compact:
                    continue
                if (now.tm_hour, now.tm_min) != COMPACT_AT or now.tm_wday != 0:
                    continue
                today = time.strftime("%Y-%m-%d")
                if self._last_compact_day == today:
                    continue
                self._last_compact_day = today
                for uid in list(self._load().keys()):
                    await self._compact(uid)
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                logger.error(f"[user_memory] 压缩定时器异常: {e}")

    # ---------- commands ----------

    @filter.command("记忆", alias={"memory"})
    async def show_memory(self, event: AstrMessageEvent):
        """查看档案：/记忆 看自己，/记忆 <QQ号> 看别人（仅管理员）"""
        uid = str(event.get_sender_id() or "")
        arg = (event.message_str or "").replace("/记忆", "").replace("/memory", "")
        arg = arg.strip()
        target = arg if arg.isdigit() else uid

        if target != uid and event.role != "admin":
            yield event.plain_result("别人的档案？先证明你是管理员")
            return

        prof = self._load().get(target) or {}
        facts = prof.get(FACTS_KEY) or []
        if not facts:
            yield event.plain_result(f"{target} 我这儿还是白纸一张")
            return
        head = f"{target} 的档案（{len(facts)} 条）"
        yield event.plain_result(head + "\n" + "\n".join(f"- {f}" for f in facts))

    @filter.command("压缩档案", alias={"compact"})
    async def compact_now(self, event: AstrMessageEvent):
        """立刻压缩自己的档案：/压缩档案"""
        uid = str(event.get_sender_id() or "")
        before = await self._compact(uid)
        if before:
            prof = self._load().get(uid) or {}
            yield event.plain_result(
                f"压缩完了，{before} 条 -> {len(prof.get(FACTS_KEY, []))} 条"
            )
        else:
            yield event.plain_result("还没到需要压缩的程度")

    @filter.command("忘记", alias={"forget"})
    async def forget_me(self, event: AstrMessageEvent):
        """清空自己的档案：/忘记"""
        uid = str(event.get_sender_id() or "")
        async with self._io_lock:
            data = self._load()
            if uid in data:
                data.pop(uid)
                self._save(data)
                removed = True
            else:
                removed = False
        if removed:
            yield event.plain_result("行，关于你的我全删了")
        else:
            yield event.plain_result("本来就没记你什么")
