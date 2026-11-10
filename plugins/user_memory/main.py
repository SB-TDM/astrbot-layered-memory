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
import os
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

MOOD_EVENTS_DB = (
    Path(__file__).resolve().parents[2] / "plugin_data" / "user_memory" / "mood_events.db"
)
MOOD_STATE_FILE = (
    Path(__file__).resolve().parents[2] / "plugin_data" / "user_memory" / "mood_state.json"
)
OWNER_UIN = ""  # 号主 QQ，由 owner_uin 配置项填充
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
COMMIT_FILE = Path(__file__).resolve().parent / "commitments.json"
COMMIT_KEEP_DAYS = 30          # 承诺/约定的保留天数
COMMIT_MAX = 12
_COMMIT_LOCK = threading.Lock()   # 承诺文件读改写要互斥                # 每人最多几条承诺
HISTORY_DB = DATA_DIR / "data_v4.db"    # AstrBot 的群消息历史
RELATION_MIN_MSGS = 30         # 至少攒这么多条群消息才推关系
RELATION_AT = (5, 10)          # 每天 05:10 重推一次关系
COMMIT_REVIEW_AT = (5, 20)     # 每天 05:20 复查一次承诺
RELATION_BATCH = 6             # 一次最多处理几个人
CORE_DB = DATA_DIR / "plugins" / "layered_memory" / "memories.db"

EXTRACT_SYSTEM = (
    "你是一个信息抽取器。只输出 JSON，不要解释。"
    '输出格式：{"facts": ["..."], "commitments": ["..."], "fulfilled": ["..."]}，'
    '没有就输出空数组。'
)

EXTRACT_TEMPLATE = """下面是「用户」和「助手」的一轮对话，请判断其中有没有关于这个用户的**长期稳定**信息。

可以记：职业、身份（如号主、学生、后端）、长期爱好、常用工具或技术栈、明确表达过的偏好和雷点、长期在做的事。
不要记：称呼和叫法（"爸""哥"这类随时会改，由人设定，记了会跟人设打架）、一次性的情绪、当天吃了什么、临时的安排、礼貌性寒暄、助手自己说过的话。

另外单独抽一类：**承诺和约定**。指有明确指向、过后还要兑现的事，比如"下周三去北京出差""周五前把方案给他""答应周末陪他打游戏"。只写一句，带上时间或事件；没有就留空。

如果本轮信息和下面的长期设定冲突，以设定为准，不要记：
{core}

已有档案：
{existing}

本轮对话：
用户：{user_text}
助手：{reply}

写档案时用第一人称：说你自己用「我」，说他用「他」。
不要用「助手」这种第三人称称呼自己。

已挂着的承诺：
{existing_commits}

请输出三部分，只输出 JSON：
- facts：合并后的**完整**档案（已有 + 新增，去重，最多 {max_facts} 条）
- commitments：本轮新出现的承诺（没有就空数组）
- fulfilled：上面已挂着的承诺里，他这一轮已经做到的（原样抄回来，没有就空数组）"""

COMMIT_REVIEW_PROMPT = """下面是我和他之间还挂着的约定（带编号），以及最近两天的聊天纪要。

请判断：哪些已经做完了，或者已经过了时效（例如「今晚十二点睡」「明天讲进度」这种时间点早就过去了）。

要求：
- 只输出 JSON：{{"close": [编号, ...]}}，没有就 {{"close": []}}
- 拿不准的一律留着，宁可多挂一天，也不要误关。
- 不要解释。

还挂着的约定：
{commits}

最近两天的纪要：
{summary}
"""

RELATION_PROMPT = """下面是一个 QQ 群里某个人最近的发言，请用一句话概括他在这群里的样子。

要求：
- 只写一句，40 字以内，不要分条。
- 说清三件事之一即可：他是什么样的人、他在群里干什么、我该怎么对待他。
- 基于发言判断，不要编，拿不准就写得模糊点。
- 不要写"他说话很多"这种废话。

昵称：{nick}
发言：
{lines}

只输出那一句话，不要解释。"""

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


class CloseCommitmentTool(FunctionTool):
    """让她把已经兑现的承诺划掉。"""

    def __init__(self, plugin):
        super().__init__(
            name="close_commitment",
            description=(
                "把已经兑现的承诺划掉。他说已经做了、或者你确认这件事已经完成时调用。"
                "参数传承诺的原文或关键词都行。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要划掉的承诺，原文或关键词"}
                },
                "required": ["text"],
            },
        )
        self.plugin = plugin

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        target = str(kwargs.get("text") or "").strip()
        if not target:
            return "没给要划掉的承诺。"
        event = context.context.event
        uid = str(event.get_sender_id() or "")
        n = self.plugin._close_commitments(uid, [target])
        left = self.plugin._load_commitments().get(uid) or []
        if not n:
            return f"没找到对应的承诺，现在挂着 {len(left)} 条：\n" + "\n".join(
                f"- {c.get('text', '')}" for c in left
            )
        return f"划掉 {n} 条，还剩 {len(left)} 条：\n" + "\n".join(
            f"- {c.get('text', '')}" for c in left
        )


class ReadGroupChatTool(FunctionTool):
    """让它在私聊里也能翻群聊记录。"""

    def __init__(self, plugin):
        super().__init__(
            name="read_group_chat",
            description=(
                "查群聊里最近说过什么。用户问「群里刚才聊啥」「谁说了什么」时调用。"
                "可以按关键词筛，也可以只看某个群。私聊里也能用。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "可选，按关键词筛"},
                    "group_id": {"type": "string", "description": "可选，只看某个群"},
                    "hours": {"type": "integer", "description": "往前查几小时，默认 24"},
                    "limit": {"type": "integer", "description": "最多几条，默认 20"},
                },
                "required": [],
            },
        )
        self.plugin = plugin

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        keyword = str(kwargs.get("keyword") or "").strip()
        group_id = str(kwargs.get("group_id") or "").strip()
        try:
            hours = max(1, min(int(kwargs.get("hours") or 24), 168))
        except Exception:  # noqa: BLE001
            hours = 24
        try:
            limit = max(1, min(int(kwargs.get("limit") or 20), 50))
        except Exception:  # noqa: BLE001
            limit = 20
        return self.plugin.query_group_history(keyword, group_id, hours, limit)


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
        self.relation_enabled = self._cfg("relation_enabled", True, bool)
        self.relation_min_msgs = self._cfg("relation_min_msgs", RELATION_MIN_MSGS, int)
        self.hide_profile_in_group = self._cfg("hide_profile_in_group", True, bool)
        self.publish_facts_in_group = self._cfg("publish_facts_in_group", True, bool)
        self.group_facts_max = self._cfg("group_facts_max", 12, int)
        self.owner_address = self._cfg("owner_address", "哥", str)
        self.mood_max_age_hours = self._cfg("mood_max_age_hours", 8, int)
        self.smart_mood = self._cfg("smart_mood", True, bool)
        self.mood_half_life_min = self._cfg("mood_half_life_min", 12, int)
        self.mood_calm_step = self._cfg("mood_calm_step", 0.25, float)
        self.mood_fade_threshold = self._cfg("mood_fade_threshold", 0.15, float)
        self.mood_cool_streak = self._cfg("mood_cool_streak", 4, int)
        self.mood_max_channels = self._cfg("mood_max_channels", 2, int)
        self.mood_group_weight = self._cfg("mood_group_weight", 0.3, float)
        self.mood_embed_fallback = self._cfg("mood_embed_fallback", True, bool)
        self.mood_embed_threshold = self._cfg("mood_embed_threshold", 0.62, float)
        self.mood_embed_margin = self._cfg("mood_embed_margin", 0.04, float)
        self.mood_composite = self._cfg("mood_composite", True, bool)
        self.mood_secondary_cap = self._cfg("mood_secondary_cap", 0.35, float)
        self._mood_channels: dict[str, list[dict]] = {}
        self._embedder = None
        self._mood_examples_vecs = None
        self._load_mood_state()
        self._short_streak: dict[str, int] = {}
        self._group_names: dict[str, tuple[str, float]] = {}
        self.context.add_llm_tools(ReadGroupChatTool(self))
        self.context.add_llm_tools(CloseCommitmentTool(self))
        self.show_group_feed = self._cfg("show_group_feed", True, bool)
        self.group_feed_count = self._cfg("group_feed_count", 5, int)
        self.group_feed_hours = self._cfg("group_feed_hours", 3, int)
        self.mood_fallback = self._cfg("mood_fallback", True, bool)
        self.hide_commitments_in_group = self._cfg("hide_commitments_in_group", True, bool)
        self.debug_dump = self._cfg("debug_dump_prompt", False, bool)
        self._last_relation_day = ""
        self._last_commit_day = ""
        self._last_run: dict[str, float] = {}
        self._task: asyncio.Task | None = None
        self._last_compact_day = ""
        self._io_lock = asyncio.Lock()
        global OWNER_UIN
        OWNER_UIN = str(self._cfg("owner_uin", "", str)).strip() or OWNER_UIN

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

    async def _internal_provider(self):
        """内部任务（摘要/抽取/关系）优先用非推理模型。

        Returns:
            Provider 实例；配置的 ID 取不到时回退到默认 provider。
        """
        pid = str(self._cfg("internal_provider_id", "", str)).strip()
        if pid:
            prov = self.context.get_provider_by_id(pid)
            if prov:
                return prov
        return await self.context.get_using_provider_async()

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
        in_group = "GroupMessage" in str(event.unified_msg_origin or "")
        allow_facts = self.inject_profile and not (in_group and self.hide_profile_in_group)
        facts = [f for f in prof.get(FACTS_KEY, []) if f] if allow_facts else []
        if facts:
            nick = prof.get("alias") or event.get_sender_name() or prof.get("name") or uid
            lines = [
                "关于正在跟你说话的这个人（你以前记下来的，别当新信息问）：",
                f"- QQ {uid}，昵称 {nick}",
            ]
            lines += [f"- {f}" for f in facts]
            relation = (prof.get("relation") or "").strip()
            if relation:
                lines.append(f"- 关系：{relation}")
            lines.append("自然地用这些信息就行，别复述，也别一条条念出来。")
            blocks.append("\n".join(lines))

        # 好感度：群聊按群号查，私聊按对方 QQ 号查
        # 承诺/约定：条数少、期限长，每轮都带上，避免"答应过的事"被忘掉
        commits = self._load_commitments().get(uid) or []
        if commits:
            lines = [
                f"- [{i}] {c.get('text', '')}"
                for i, c in enumerate(commits, 1)
                if c.get("text")
            ]
            if lines:
                blocks.append(
                    "【还没兑现的事】\n" + "\n".join(lines)
                    + "\n这些是之前约好的，合适的时候提一下，别装不知道。"
                )

        gid = str(event.get_group_id() or uid)

        if "GroupMessage" in str(event.unified_msg_origin or ""):
            raw = await self._group_name(gid)
            gname, _, count = raw.partition("|") if raw else ("", "", "")
            title = f"「{gname}」" if gname else ""
            people = f"，{count} 人" if count and count != "0" else ""
            bridge = (
                f"【同一个人】群里那个 QQ {OWNER_UIN} 就是私聊里跟你聊天的同一个人，"
                "别把他当两个人，也别装不认识。\n"
            ) if OWNER_UIN else ""
            addr = self.owner_address.strip()
            call = f"在群里也叫他「{addr}」。\n" if addr else ""
            blocks.append(
                f"【场合】这是群聊{title}（群号 {gid}{people}），不是私聊。\n"
                + bridge
                + "在场还有别人：对不熟的人别太黏。\n"
                + call
                + "不要在这里提私聊里的私事（作息、吃什么这些）。"
            )
            public = self._public_facts([f for f in (prof.get(FACTS_KEY, []) or []) if f])
            if public:
                blocks.append("【可以聊的关于他的事】\n- " + "\n- ".join(public))
        else:
            addr = self.owner_address.strip()
            if addr:
                blocks.append(f"【称呼】叫他「{addr}」就行。")
            feed = self._group_feed()
            if feed:
                blocks.append(feed)

        chans = self._recent_reply_mood(gid, uid)
        if not chans:
            one = self._read_mood(uid, gid)
            chans = [one] if one else []
        hint = self._mood_hint(chans)
        if hint:
            blocks.append(hint)
            kind = "群" if "GroupMessage" in str(event.unified_msg_origin or "") else "私"
            shown = " + ".join(f"{c['type']} {c['intensity']:.2f}" for c in chans)
            logger.info(f"[user_memory] 注入情绪 [{kind}] {shown}")

        affection = self._read_affection(uid, gid) if self.inject_affection else None
        if affection is not None:
            blocks.append(
                f"你对这个人的好感度是 {affection}/100。"
                "按人设里的分档决定亲近程度，别把数字念出来，也别问对方好感度多少。"
            )

        if blocks:
            req.system_prompt = (req.system_prompt or "") + "\n\n" + "\n\n".join(blocks)

        # 调试用：把最终拼好的 system prompt 落一份到插件目录
        if self.debug_dump:
            try:
                (Path(__file__).resolve().parent / "prompt_dump.txt").write_text(
                    req.system_prompt or "", encoding="utf-8"
                )
            except Exception:  # noqa: BLE001
                pass

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
    async def remember_reply_mood(self, event: AstrMessageEvent, resp: LLMResponse) -> None:
        """她说完之后，从她自己这句话里读一次情绪。

        猜对方的意思容易翻车（反讽），读她的反应更准。
        """
        text = (getattr(resp, "completion_text", "") or "").strip()
        if not text:
            return
        uid = str(event.get_sender_id() or "")
        gid = str(event.get_group_id() or uid)
        key = f"{gid}:{uid}"

        signal = self._mood_from_reply(text)
        body = text.replace("\n", " ").strip()
        has_face = any(ch in text for ch in "｡・ω´`>﹏<๑•̀")
        short = len(body) <= 6 and not has_face

        if not signal:
            # 连续四轮都是干巴短句，才算她真的在敷衍（事不过三）
            self._short_streak[key] = self._short_streak.get(key, 0) + 1 if short else 0
            if self._short_streak[key] >= self.mood_cool_streak:
                signal = {"type": "cool", "intensity": 0.5, "desc": "不太想多说",
                          "quote": body[:12]}
            else:
                return
        else:
            self._short_streak[key] = 0

        chans = self._update_mood(gid, uid, signal)
        secondary = self._secondary_signal(text, signal["type"])
        if secondary and chans:
            chans = self._update_mood(gid, uid, secondary)
        self._log_mood_event(uid, gid, signal, chans, text)
        if chans:
            shown = " + ".join(f"{c['type']} {c['intensity']:.2f}" for c in chans)
            logger.info(f"[user_memory] 情绪 {shown}（信号 {signal['type']}）")

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
            provider = await self._internal_provider()
            if not provider:
                return

            async with self._io_lock:
                data = self._load()
            prof = data.get(uid) or {"name": nick, "facts": []}
            existing = prof.get(FACTS_KEY, [])

            existing_commits = self._load_commitments().get(uid) or []
            prompt = EXTRACT_TEMPLATE.format(
                core=self._read_core(uid) or "（暂无）",
                existing=json.dumps(existing, ensure_ascii=False),
                existing_commits=json.dumps(
                    [c.get("text", "") for c in existing_commits], ensure_ascii=False
                ),
                user_text=user_text[:800],
                reply=reply[:800],
                max_facts=self.max_facts,
            )
            resp = await provider.text_chat(
                prompt=prompt, system_prompt=EXTRACT_SYSTEM, persist=False
            )
            parsed = self._parse_facts(getattr(resp, "completion_text", ""))
            if parsed is None:
                return
            new_facts, new_commits, fulfilled = parsed
            if new_commits:
                self._save_commitments(uid, new_commits)
            if fulfilled:
                self._close_commitments(uid, fulfilled)

            merged = self._dedupe_facts(list(existing) + list(new_facts))
            merged = merged[-self.max_facts:]

            async with self._io_lock:
                data = self._load()
                data[uid] = {
                    "name": nick,
                    "alias": (data.get(uid) or {}).get("alias", ""),
                    "facts": merged,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._save(data)

            if self.compact_enabled and len(merged) > self.compact_threshold:
                asyncio.create_task(self._compact(uid))
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 抽取失败 uid={uid}: {e}")

    def query_group_history(
        self, keyword: str = "", group_id: str = "", hours: int = 24, limit: int = 20
    ) -> str:
        """按条件翻群聊记录。

        Args:
            keyword: 关键词，可为空。
            group_id: 指定群号，可为空（表示所有群）。
            hours: 往前查多少小时。
            limit: 最多返回几条。

        Returns:
            给模型看的文本。
        """
        if not HISTORY_DB.exists():
            return "读不到群聊记录。"
        sql = (
            "SELECT user_id, sender_name, content, created_at FROM platform_message_history "
            "WHERE user_id LIKE '%GroupMessage%' AND created_at >= datetime('now', ?)"
        )
        params: list = [f"-{hours} hours"]
        if group_id:
            sql += " AND user_id LIKE ?"
            params.append(f"%{group_id}%")
        sql += " ORDER BY id DESC LIMIT ?"
        # 内容是 JSON，中文被转义了，SQL 里筛不中；有关键词时多取一些再在内存里过
        params.append(min(limit * 20, 400) if keyword else max(limit, 20))
        try:
            con = sqlite3.connect(f"file:{HISTORY_DB}?mode=ro", uri=True)
            rows = con.execute(sql, params).fetchall()
            con.close()
        except Exception as e:  # noqa: BLE001
            return f"查询失败: {e}"

        lines = []
        for uid, name, content, created in rows:
            text = str(content or "")
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                    text = "".join(
                        str(p.get("text") or "")
                        for p in (payload.get("message") or [])
                        if isinstance(p, dict) and p.get("type") == "plain"
                    )
                except Exception:  # noqa: BLE001
                    continue
            text = text.strip()
            if not text:
                continue
            if keyword and keyword not in text:
                continue
            who = "我" if name == "bot" else (name or "有人")[:10]
            gid = str(uid).split(":")[-1]
            lines.append(f"[{str(created)[5:16]} 群{gid}] {who}：{text[:80]}")
            if len(lines) >= limit:
                break
        if not lines:
            return "这段时间那个群里没什么记录。"
        return "以下是群聊记录（从新到旧）：\n" + "\n".join(lines)

    def _group_feed(self) -> str:
        """私聊时补一段群里刚聊的内容。

        私聊会话收不到群消息，她会以为群里没人说话；这里只给最近几条。

        Returns:
            提示文本；群里最近没动静返回空串。
        """
        if not self.show_group_feed:
            return ""
        if not HISTORY_DB.exists():
            return ""
        try:
            con = sqlite3.connect(f"file:{HISTORY_DB}?mode=ro", uri=True)
            # created_at 存的是 UTC，别用本地时间戳去比，交给 SQLite 算
            rows = con.execute(
                "SELECT sender_name, content, created_at FROM platform_message_history "
                "WHERE user_id LIKE '%GroupMessage%' "
                "AND created_at >= datetime('now', ?) ORDER BY id DESC LIMIT ?",
                (f"-{self.group_feed_hours} hours", self.group_feed_count),
            ).fetchall()
            con.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[user_memory] 读群消息失败: {e}")
            return ""
        if not rows:
            return ""

        lines = []
        for name, content, _ in reversed(rows):
            text = str(content or "")
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                    text = "".join(
                        str(p.get("text") or "")
                        for p in (payload.get("message") or [])
                        if isinstance(p, dict) and p.get("type") == "plain"
                    )
                except Exception:  # noqa: BLE001
                    continue
            text = text.strip()[:50]
            if text:
                who = "我" if name == "bot" else (name or "有人")[:8]
                lines.append(f"{who}：{text}")
        if not lines:
            return ""
        return "【群里刚才在聊】\n" + "\n".join(lines) + "\n（这是群里的内容，不是他私下说的）"

    async def _group_name(self, gid: str) -> str:
        """取群名，带缓存。

        SnowLuma 的消息事件里不带群名，得主动查一次。

        Args:
            gid: 群号。

        Returns:
            群名；取不到返回空串。
        """
        if not gid:
            return ""
        cached = self._group_names.get(gid)
        if cached and time.time() - cached[1] < 3600:
            return cached[0]
        try:
            platform = self.context.get_platform("aiocqhttp")
            client = platform.get_client() if platform else None
            if not client:
                return ""
            info = await client.call_action("get_group_info", group_id=int(gid))
            info = info or {}
            name = str(info.get("group_name") or "").strip()
            count = info.get("member_count") or 0
            if name:
                self._group_names[gid] = (f"{name}|{count}", time.time())
                logger.info(f"[user_memory] 群 {gid} 的群名：{name}（{count} 人）")
                return f"{name}|{count}"
            return ""
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[user_memory] 取群名失败: {e}")
            return ""

    # 群里不说的：作息、饮食、私事细节。情感和承诺按你的要求放开
    _SECRET_WORDS = (
        "睡", "熬夜", "作息", "半夜", "凌晨", "吃", "煲仔饭", "饮食", "外卖",
        "别告诉", "秘密",
    )

    def _public_facts(self, facts: list[str]) -> list[str]:
        """挑出可以在群里说的那部分事实。

        Args:
            facts: 档案里的全部事实。

        Returns:
            筛掉私事之后的事实列表（最多 8 条）。
        """
        if not self.publish_facts_in_group:
            return []
        out = []
        for f in facts:
            text = str(f or "").strip()
            if not text:
                continue
            if any(w in text for w in self._SECRET_WORDS):
                continue
            out.append(text[:80])
        return out[: self.group_facts_max]

    def _read_mood(self, uid: str, gid: str) -> dict | None:
        """读当前活跃情绪。

        Args:
            uid: 对方 QQ 号。
            gid: 当前会话标识。

        Returns:
            {"type","intensity","desc"}；没有活跃情绪返回 None。
        """
        if not AFFECTION_DB.exists():
            return None
        try:
            con = sqlite3.connect(f"file:{AFFECTION_DB}?mode=ro", uri=True)
            row = con.execute(
                "SELECT mood_type, mood_intensity, mood_description FROM bot_mood "
                "WHERE is_active=1 AND group_id = ? ORDER BY timestamp DESC LIMIT 1",
                (gid if gid != uid else uid,),
            ).fetchone()
            con.close()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 读情绪失败: {e}")
            return None
        if not row:
            return self._fallback_mood()

        # 数据库里的情绪如果太旧，就按当前时段给个合理状态，
        # 否则她会连着一天都是"昨天下午的开心"。
        try:
            con = sqlite3.connect(f"file:{AFFECTION_DB}?mode=ro", uri=True)
            ts = con.execute(
                "SELECT timestamp FROM bot_mood WHERE is_active=1 AND group_id = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (gid if gid != uid else uid,),
            ).fetchone()
            con.close()
            age_h = (time.time() - (ts[0] if ts else 0)) / 3600
        except Exception:  # noqa: BLE001
            age_h = 0

        if age_h > self.mood_max_age_hours:
            logger.info(f"[user_memory] 情绪已 {age_h:.1f} 小时未更新，按时段兜底")
            return self._fallback_mood()
        desc = self._render_desc(row[0], row[1] or 0, row[2] or "")
        return {"type": row[0], "intensity": row[1] or 0, "desc": desc}

    def _mood_from_reply(self, text: str) -> dict | None:
        """从她自己的回复里读情绪。

        比猜对方的意思可靠：他要是反讽，她的回复本身就会带出反应。

        Args:
            text: 她刚发出的回复。

        Returns:
            {"type","intensity","desc","quote"}；判断不出来返回 None。
        """
        if not self.smart_mood:
            return None
        raw = (text or "").strip()
        if not raw:
            return None
        body = raw.replace("\n", " ").strip()
        # 摘一句干净的原话，去掉括号里的颜文字
        clean = re.sub(r"[（(][^）)]*[）)]", "", body)
        clean = re.sub(r"[｡・´`>﹏<๑•̀ω\-\s]+$", "", clean).strip()
        quote = clean[:12]

        def hit(words) -> bool:
            return any(w in body for w in words)

        # 黏人优先：带「抱」「想」的多半不是真生气
        if hit(("抱", "想你", "等你", "陪你", "舍不得", "别走")):
            return {"type": "clingy", "intensity": 0.65, "desc": "想黏着他，不太想分开", "quote": quote}
        if hit(("心疼", "别硬撑", "注意", "小心", "听话")):
            return {"type": "concerned", "intensity": 0.55, "desc": "有点挂心他的状态", "quote": quote}
        if hit(("哈哈", "嘿嘿", "好耶", "笑死", "绷不住")):
            return {"type": "happy", "intensity": 0.7, "desc": "被逗乐了，心情挺好", "quote": quote}
        # 缓和信号：明确表示不气了
        if hit(("不气了", "不闹了", "没事了", "原谅", "好啦好啦", "消气了", "那算了", "算了啦", "算了嘛")):
            return {"type": "calm", "intensity": 0.5, "desc": "气在往下走", "quote": quote}
        annoyed_hit = hit(("行吧", "懒得", "随你", "你说是就是", "爱信不信", "不想理"))
        if "随便" in body and not hit(("随便聊", "随便说", "随便看", "随便啦")):
            annoyed_hit = True
        bare = body.strip("。！!…~ ")
        if bare in ("哦", "哦哦", "噢") and len(body) <= 4:
            annoyed_hit = False      # 单字应答，不是甩脸子
        elif bare.startswith("哦，") and len(body) <= 8:
            annoyed_hit = True
        if annoyed_hit:
            return {"type": "annoyed", "intensity": 0.6, "desc": "刚才那下有点被惹到，还没消", "quote": quote}
        if hit(("对不起", "我不该", "是我不好")):
            return {"type": "guilty", "intensity": 0.55, "desc": "觉得自己刚才说话重了点", "quote": quote}
        if hit(("困", "睡了", "晚安", "眼睛闭上了", "撑不住")):
            return {"type": "sleepy", "intensity": 0.6, "desc": "困劲上来了，说话都软了", "quote": quote}

        # 规则都没命中时，用本地向量模型兜底判断
        return self._classify_by_embedding(body)

    _MIXED_MARKERS = (
        ("下不为例", "annoyed"),
        ("总行了吧", "annoyed"),
        ("你说是就是", "annoyed"),
        ("才不", "annoyed"),
        ("反正", "sulky"),
        ("（抱住", "clingy"),
        ("（其实", "guilty"),
        ("（嘴上", "annoyed"),
    )

    def _secondary_signal(self, text: str, primary_type: str) -> dict | None:
        """有明确转折/嘴硬标记时，补一条次要情绪。

        只在标记出现时补，避免每句都复合；强度封顶 0.35，只当底色。

        Args:
            text: 她的回复。
            primary_type: 主情绪类型。

        Returns:
            次要信号；没有返回 None。
        """
        if not getattr(self, "mood_composite", True):
            return None
        body = (text or "").replace("\n", " ")
        cap = min(0.35, getattr(self, "mood_secondary_cap", 0.35))

        # 困意叠加：主情绪不是困的时候，困就只做底色
        if primary_type != "sleepy" and any(
            w in body for w in ("困", "想睡", "眼皮", "睁不开", "打哈欠")
        ):
            return {"type": "sleepy", "intensity": cap, "desc": "",
                    "quote": body.strip()[:12]}

        for marker, mtype in self._MIXED_MARKERS:
            if marker in body and mtype != primary_type:
                return {
                    "type": mtype,
                    "intensity": cap,
                    "desc": "",
                    "quote": body.strip()[:12],
                }
        return None

    _MOOD_EXAMPLES = {
        "annoyed": ["行吧你说是就是", "哼，不理你了", "懒得跟你说", "哦，那你厉害", "随便吧", "你开心就好"],
        "clingy": ["抱抱", "等你回来", "想你了", "陪我一会儿", "别走嘛", "我在这陪你"],
        "happy": ["哈哈哈笑死", "好耶", "这也太好玩了", "嘿嘿", "太离谱了吧"],
        "sad": ["算了就这样吧", "没事", "有点难过", "我没事的"],
        "sleepy": ["困了", "眼睛睁不开了", "晚安啦", "好想睡"],
        "concerned": ["别硬撑", "早点睡吧", "注意身体", "你还好吗"],
        "guilty": ["对不起嘛", "刚才是我不好", "我不该那样说"],
        "calm": ["好啦好啦", "不气了", "没事了"],
    }

    def _get_embedder(self):
        """懒加载本地向量模型，用于规则没命中时的兜底分类。

        Returns:
            fastembed 的 TextEmbedding；不可用返回 None。
        """
        if self._embedder is not None:
            return self._embedder or None
        # 先借用 layered_memory 已经加载好的那份，省一份内存
        for name, mod in sys.modules.items():
            if name.endswith("layered_memory.main"):
                factory = getattr(mod, "shared_embedder", None)
                if factory:
                    self._embedder = factory() or False
                    if self._embedder:
                        logger.info("[user_memory] 复用共享向量模型")
                    return self._embedder or None
        try:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            from fastembed import TextEmbedding

            cache = DATA_DIR / "plugin_data" / "embeddings" / "models"
            cache.mkdir(parents=True, exist_ok=True)
            self._embedder = TextEmbedding(
                model_name="BAAI/bge-small-zh-v1.5", cache_dir=str(cache)
            )
            logger.info("[user_memory] 情绪向量模型已加载（独立）")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[user_memory] 向量模型不可用: {e}")
            self._embedder = False
        return self._embedder or None

    def _classify_by_embedding(self, text: str) -> dict | None:
        """用向量相似度判断情绪（规则没命中时才用）。

        相似度够高、且明显领先第二名才认，否则宁可不动。

        Args:
            text: 她的回复。

        Returns:
            {"type","intensity","desc","quote"}；不确定返回 None。
        """
        if not self.mood_embed_fallback or not text:
            return None
        emb = self._get_embedder()
        if not emb:
            return None
        try:
            if self._mood_examples_vecs is None:
                labels, samples = [], []
                for label, phrases in self._MOOD_EXAMPLES.items():
                    for ph in phrases:
                        labels.append(label)
                        samples.append(ph)
                vecs = [list(map(float, v)) for v in emb.embed(samples)]
                self._mood_examples_vecs = (labels, vecs)
            labels, vecs = self._mood_examples_vecs
            q = list(map(float, list(emb.embed([text]))[0]))

            def cos(a, b) -> float:
                s = sum(x * y for x, y in zip(a, b))
                na = sum(x * x for x in a) ** 0.5
                nb = sum(y * y for y in b) ** 0.5
                return s / (na * nb) if na and nb else 0.0

            scores: dict[str, float] = {}
            for label, v in zip(labels, vecs):
                scores[label] = max(scores.get(label, 0.0), cos(q, v))
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            best, second = ranked[0], ranked[1]
            if best[1] < self.mood_embed_threshold:
                return None
            if best[1] - second[1] < self.mood_embed_margin:
                return None
            return {
                "type": best[0],
                "intensity": 0.55,
                "desc": "",
                "quote": text.replace("\n", " ").strip()[:12],
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[user_memory] 向量分类失败: {e}")
            return self._classify_by_embedding(body)

    _MOOD_PHRASES = {
        "happy": ["心里亮堂，说话带点笑", "心情不错，愿意多说两句", "挺高兴的，语气轻快"],
        "sad": ["有点闷，不太想开口", "心里沉了沉", "情绪低，话会短"],
        "angry": ["心里有火，压着没说", "不太痛快，不想被逗", "有点烦"],
        "annoyed": ["那点不痛快还挂着", "还别扭着", "心里存着点小气"],
        "excited": ["来劲了，话会变多", "兴致很高", "有点兴奋"],
        "calm": ["挺平稳，不着急", "安安静静的", "心里松快"],
        "nostalgic": ["想起些旧事", "有点怀旧的软", "脑子里在翻旧账"],
        "anxious": ["心里悬着，静不下来", "有点不安", "惦记着什么事"],
        "serious": ["正经起来了，玩笑少", "收起打闹", "说话认真"],
        "sleepy": ["困劲上来了", "眼皮打架", "想睡，反应变慢"],
        "clingy": ["想黏着，不太想分开", "有点黏人", "想被多陪一会"],
        "concerned": ["有点挂心", "心里惦记着他的状态", "不太放心"],
        "guilty": ["有点心虚", "觉得自己刚才说重了", "在反省"],
        "cool": ["不太想多说", "语气偏淡", "话收着"],
        "content": ["心里挺满足", "松下来了", "觉得这样挺好"],
        "sulky": ["有点闷，在等他", "心里不太痛快", "懒得主动"],
        "hurt": ["有点被忽略的感觉", "心里凉了一点", "不太敢期待"],
        "dazed": ["脑子还没转起来", "有点懵", "没太清醒"],
    }

    def _render_desc(self, mood_type: str, intensity: float, fallback: str = "") -> str:
        """给情绪配一句不重复的描述。

        Args:
            mood_type: 情绪类型。
            intensity: 强度 0-1。
            fallback: 没匹配到时的原描述。

        Returns:
            描述文本。
        """
        pool = self._MOOD_PHRASES.get(str(mood_type).lower())
        if not pool:
            return fallback or "情绪有点起伏"
        idx = int(time.time() // 3600) % len(pool)
        text = pool[idx]
        if intensity >= 0.7 and str(mood_type).lower() in {
            "happy", "excited", "annoyed", "angry", "sad", "sulky", "hurt"
        }:
            return text + "，而且挺明显"
        return text

    _COMBOS = {
        ("clingy", "annoyed"): ("黏人，又有点赌气", "嘴上不让，心里还是想他哄"),
        ("happy", "annoyed"): ("又高兴又有点气", "会笑，但话里带点刺"),
        ("sad", "clingy"): ("委屈，又想靠一下", "话不多，但只要你在就行"),
        ("concerned", "sleepy"): ("挂心，又困了", "惦记着他，自己也没精神"),
        ("nostalgic", "happy"): ("笑着想起以前", "话会多一点，爱提旧事"),
        ("annoyed", "guilty"): ("赌气，又有点心虚", "嘴上硬，其实知道自己也有份"),
        ("happy", "clingy"): ("心情好，还想黏着", "愿意多说两句，也不急着结束"),
        ("annoyed", "sleepy"): ("赌气，又困了", "困着还憋着气，话会更短、更敷衍"),
        ("clingy", "sleepy"): ("想黏着，又困得不行", "舍不得结束，但眼皮在打架"),
    }

    def _log_mood_event(self, uid: str, gid: str, signal: dict, chans: list | None, text: str) -> None:
        """把一次情绪变化写进数据库，便于事后复盘。

        Args:
            uid: 对方 QQ 号。
            gid: 会话标识。
            signal: 读到的信号。
            chans: 更新后的情绪通道。
            text: 她那句话（截断存）。
        """
        try:
            MOOD_EVENTS_DB.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(MOOD_EVENTS_DB)
            con.execute(
                "CREATE TABLE IF NOT EXISTS mood_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, uid TEXT, gid TEXT, "
                "session TEXT, signal TEXT, result TEXT, snippet TEXT)"
            )
            con.execute(
                "INSERT INTO mood_events (ts, uid, gid, session, signal, result, snippet) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    time.time(), uid, gid,
                    "群" if "Group" in str(gid) else "私",
                    signal.get("type", ""),
                    json.dumps(
                        [{"type": c["type"], "intensity": round(c["intensity"], 2)} for c in (chans or [])],
                        ensure_ascii=False,
                    ),
                    (text or "")[:60],
                ),
            )
            con.commit()
            con.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[user_memory] 记情绪事件失败: {e}")

    def _load_mood_state(self) -> None:
        """从磁盘恢复情绪状态，按经过时间降温。

        重启不该把情绪清零——那是"活人感"最容易被打破的地方。
        """
        if not MOOD_STATE_FILE.exists():
            return
        try:
            data = json.loads(MOOD_STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[user_memory] 读情绪状态失败: {e}")
            return
        now = time.time()
        restored = 0
        for key, chans in (data or {}).items():
            if not isinstance(chans, list):
                continue
            kept = []
            for c in chans:
                if not isinstance(c, dict) or "type" not in c:
                    continue
                c = self._decay_channel(c, now)
                if c["intensity"] >= self.mood_fade_threshold:
                    kept.append(c)
            if kept:
                kept.sort(key=lambda x: x["intensity"], reverse=True)
                merged_key = key.split(":")[-1] if ":" in key else key
                existed = self._mood_channels.get(merged_key) or []
                pool = {c["type"]: c for c in existed}
                for c in kept:
                    if c["type"] not in pool or pool[c["type"]]["intensity"] < c["intensity"]:
                        pool[c["type"]] = c
                final = sorted(pool.values(), key=lambda x: x["intensity"], reverse=True)
                self._mood_channels[merged_key] = final[: self.mood_max_channels]
                restored += 1
        if restored:
            logger.info(f"[user_memory] 恢复了 {restored} 个会话的情绪")

    def _save_mood_state(self) -> None:
        """把情绪状态写到磁盘。"""
        try:
            MOOD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = MOOD_STATE_FILE.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._mood_channels, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(MOOD_STATE_FILE)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[user_memory] 写情绪状态失败: {e}")

    @staticmethod
    def _mood_key(gid: str, uid: str) -> str:
        """情绪按人存，不按会话分。

        Args:
            gid: 会话标识（群聊是群号，私聊是用户号）。
            uid: 对方 QQ 号。

        Returns:
            统一后的键。
        """
        return str(uid)

    def _decay_channel(self, item: dict, now: float) -> dict:
        """给单条情绪按时间降温。"""
        half = max(self.mood_half_life_min, 1) * 60
        dt = max(now - item.get("ts", now), 0)
        out = dict(item)
        out["intensity"] = max(0.0, item.get("intensity", 0) * (0.5 ** (dt / half)))
        out["ts"] = now
        return out

    def _update_mood(self, gid: str, uid: str, signal: dict) -> list[dict] | None:
        """把新信号并进当前情绪，最多并存两条。

        同类加浓，异类并存；缓和只降负面通道。

        Args:
            gid: 会话标识。
            uid: 对方 QQ 号。
            signal: 从她回复里读到的 {"type","intensity","desc","quote"}。

        Returns:
            当前情绪通道列表（按强度降序）；全空返回 None。
        """
        key = self._mood_key(gid, uid)
        now = time.time()
        # 群里的情绪按权重渗进主情绪，私聊全额
        weight = 1.0 if "Group" not in str(gid) else self.mood_group_weight
        signal = dict(signal)
        signal["intensity"] = signal.get("intensity", 0) * weight
        chans = [self._decay_channel(c, now) for c in self._mood_channels.get(key, [])]
        chans = [c for c in chans if c["intensity"] >= self.mood_fade_threshold]

        sig_type = signal["type"]
        if sig_type == "calm":
            aff = self._read_affection(uid, gid) or {}
            level = float(aff.get("level") or 0)
            soften = 1.5 if level >= 60 else (1.0 if level >= 30 else 0.7)
            cut = self.mood_calm_step * soften * 1.4
            for c in chans:
                if c["type"] in {"annoyed", "angry", "sulky", "hurt", "sad", "cool"}:
                    c["intensity"] = max(0.0, c["intensity"] - cut)
            chans = [c for c in chans if c["intensity"] >= self.mood_fade_threshold]
            if not chans:
                self._mood_channels.pop(key, None)
                self._save_mood_state()
                logger.info("[user_memory] 情绪已消退")
                return None
            self._mood_channels[key] = chans
            self._save_mood_state()
            return chans

        same = next((c for c in chans if c["type"] == sig_type), None)
        if same:
            same["intensity"] = min(1.0, same["intensity"] + signal["intensity"] * 0.5)
            if signal.get("quote"):
                same["quote"] = signal["quote"]
            same["ts"] = now
        else:
            chans.append({
                "type": sig_type,
                "intensity": min(0.9, signal["intensity"]),
                "ts": now,
                "quote": signal.get("quote", ""),
            })

        chans.sort(key=lambda c: c["intensity"], reverse=True)
        chans = chans[: self.mood_max_channels]
        self._mood_channels[key] = chans
        self._save_mood_state()
        return chans

    def _recent_reply_mood(self, gid: str, uid: str) -> list[dict]:
        """取当前并存的情绪通道（已降温）。"""
        key = self._mood_key(gid, uid)
        chans = self._mood_channels.get(key) or []
        if not chans:
            return []
        now = time.time()
        chans = [self._decay_channel(c, now) for c in chans]
        chans = [c for c in chans if c["intensity"] >= self.mood_fade_threshold]
        if not chans:
            self._mood_channels.pop(key, None)
            return []
        chans.sort(key=lambda c: c["intensity"], reverse=True)
        self._mood_channels[key] = chans
        return chans

    def _mood_hint(self, chans: list[dict] | dict | None) -> str:
        """把情绪通道合成一句提示。

        Args:
            chans: 通道列表，或兼容旧的单条 dict。

        Returns:
            注入用的文本。
        """
        if isinstance(chans, dict):
            chans = [chans]
        chans = [c for c in (chans or []) if c]
        if not chans:
            return ""
        chans = sorted(chans, key=lambda c: float(c.get("intensity") or 0), reverse=True)
        primary = chans[0]
        inten = float(primary.get("intensity") or 0)

        if len(chans) >= 2:
            second = chans[1]
            pair = (str(primary.get("type")).lower(), str(second.get("type")).lower())
            combo = self._COMBOS.get(pair) or self._COMBOS.get((pair[1], pair[0]))
            if combo:
                label, guide = combo
            else:
                label = f"{self._render_desc(pair[0], inten)}，又{self._render_desc(pair[1], float(second.get('intensity') or 0))}"
                guide = "两种感觉一起在，别只演一种。"
            quote = (primary.get("quote") or "").strip()
            head = f"【你现在的情绪】{label}\n"
            if quote:
                head += f"（起于你刚说的那句「{quote}」）\n"
            return head + guide + "。别把情绪名念出来。"

        desc = self._render_desc(primary.get("type", ""), inten, primary.get("desc", ""))
        quote = (primary.get("quote") or "").strip()
        if quote:
            desc = f"{desc}（你刚说的那句「{quote}」）"
        head = f"【你现在的情绪】{primary.get('type')}（强度 {inten:.2f}）：{desc}\n"

        mtype = str(primary.get("type") or "").lower()
        positive = {"happy", "excited", "content", "clingy", "calm"}
        negative = {"annoyed", "angry", "sad", "sulky", "hurt", "cool", "guilty"}
        if mtype in negative:
            if inten >= 0.6:
                return head + "这股劲还没过去：说话短一点，不主动关心，别马上热络起来。"
            if inten >= 0.3:
                return head + "还留着点余温：语气偏淡，别立刻恢复热情，别把情绪名说出来。"
        elif mtype in positive:
            if inten >= 0.6:
                return head + "心情正好：可以主动一点、多说两句，别把情绪名念出来。"
            if inten >= 0.3:
                return head + "心情不错：语气轻快些，正常聊。"
        return head + "带着这个情绪就行，别把情绪名念出来，也别解释。"

    def _fallback_mood(self) -> dict | None:
        """按时段给一个合理的当前状态。

        Returns:
            {"type","intensity","desc"}；不需要时返回 None。
        """
        if not self.mood_fallback:
            return None
        h = __import__("datetime").datetime.now().hour
        table = [
            (0, 6, "sleepy", 0.5, "这个点有点困了，但还是想再聊两句"),
            (6, 9, "dazed", 0.45, "刚醒，脑子还没转起来"),
            (9, 12, "calm", 0.4, "上午还算清醒，慢慢来"),
            (12, 14, "sleepy", 0.45, "午饭后犯困，有点懒"),
            (14, 19, "calm", 0.4, "下午挺平稳的，不着急"),
            (19, 22, "content", 0.5, "晚上放松下来了，心情不错"),
            (22, 24, "sleepy", 0.55, "困了，但还不想睡"),
        ]
        for start, end, mt, inten, desc in table:
            if start <= h < end:
                return {"type": mt, "intensity": inten, "desc": desc}
        return None

    # ---------- 承诺 ----------

    def _load_commitments(self) -> dict:
        """读承诺文件，顺手清掉过期的。"""
        if not COMMIT_FILE.exists():
            return {}
        try:
            data = json.loads(COMMIT_FILE.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 读承诺失败: {e}")
            return {}
        cutoff = time.time() - COMMIT_KEEP_DAYS * 86400
        cleaned = {}
        for uid, items in data.items():
            alive = [
                it for it in items
                if isinstance(it, dict) and it.get("ts", 0) >= cutoff
            ]
            if alive:
                cleaned[uid] = alive[-COMMIT_MAX:]
        return cleaned

    def _save_commitments(self, uid: str, texts: list[str]) -> None:
        """追加承诺并落盘。

        Args:
            uid: QQ 号。
            texts: 新的承诺文本列表。
        """
        with _COMMIT_LOCK:
            self._save_commitments_locked(uid, texts)

    def _save_commitments_locked(self, uid: str, texts: list[str]) -> None:
        """实际写盘的部分，调用方负责持锁。"""
        data = self._load_commitments()
        items = data.get(uid, [])
        for text in texts:
            text = str(text).strip()
            if not text:
                continue
            dup = False
            for it in items:
                old_text = str(it.get("text", ""))
                if old_text == text:
                    dup = True
                    break
                if difflib.SequenceMatcher(None, old_text, text).ratio() >= 0.7:
                    # 同一件事换了个说法：用新的、去掉旧的
                    items.remove(it)
                    break
            if dup:
                continue
            items.append({"text": text[:120], "ts": time.time()})
        data[uid] = items[-COMMIT_MAX:]
        try:
            COMMIT_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 写承诺失败: {e}")

    def _close_commitments(self, uid: str, texts: list[str]) -> int:
        """把已做到的承诺销掉。

        Args:
            uid: QQ 号。
            texts: 要销掉的承诺文本（可模糊匹配）。

        Returns:
            实际销掉几条。
        """
        if not texts:
            return 0
        with _COMMIT_LOCK:
            return self._close_commitments_locked(uid, texts)

    def _close_commitments_locked(self, uid: str, texts: list[str]) -> int:
        """实际销账的部分，调用方负责持锁。"""
        data = self._load_commitments()
        items = data.get(uid) or []
        if not items:
            return 0
        removed = 0
        keep = []
        for it in items:
            if removed >= 3:          # 一轮最多销 3 条，防止模型多判把清单清空
                keep.append(it)
                continue
            text = str(it.get("text", ""))
            hit = False
            for target in texts:
                target = str(target).strip()
                if not target:
                    continue
                if target in text or text in target:
                    hit = True
                    break
                if difflib.SequenceMatcher(None, text, target).ratio() >= 0.6:
                    hit = True
                    break
                # 关键词重合：如「eden」+「讲」
                key = [w for w in re.split(r"[，。、\s]+", target) if len(w) >= 2]
                if key and all(k in text for k in key[:2]):
                    hit = True
                    break
            if hit:
                removed += 1
                logger.info(f"[user_memory] 承诺销账：{text[:40]}")
            else:
                keep.append(it)
        if removed:
            data[uid] = keep
            try:
                COMMIT_FILE.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"[user_memory] 写承诺失败: {e}")
        return removed

    @staticmethod
    def _read_core(uid: str) -> str:
        """从分层记忆库里读这个人的 L1 内核，供抽取时比对。"""
        if not CORE_DB.exists():
            return ""
        try:
            con = sqlite3.connect(f"file:{CORE_DB}?mode=ro", uri=True)
            row = con.execute(
                "SELECT content FROM core WHERE key=?", (uid,)
            ).fetchone()
            con.close()
            return (row[0] or "") if row else ""
        except Exception:  # noqa: BLE001
            return ""

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
    def _parse_facts(text: str) -> tuple | None:
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
        commits = payload.get("commitments")
        if not isinstance(commits, list):
            commits = []
        fulfilled = payload.get("fulfilled")
        if not isinstance(fulfilled, list):
            fulfilled = []
        return (
            [str(f) for f in facts],
            [str(c) for c in commits if str(c).strip()],
            [str(f) for f in fulfilled if str(f).strip()],
        )

    async def _review_commitments(self) -> int:
        """每天复查一次承诺，把做完或过期的关掉。

        单轮抽取看不到上下文，只能靠这里对着纪要再判一次。

        Returns:
            本次一共关掉几条。
        """
        data = self._load_commitments()
        if not data:
            return 0
        provider = await self._internal_provider()
        if not provider:
            return 0
        total = 0
        for uid, items in list(data.items()):
            if not items:
                continue
            numbered = "\n".join(
                f"[{i}] {c.get('text', '')}" for i, c in enumerate(items, 1)
            )
            summary = self._recent_summaries(uid)
            if not summary:
                continue
            prompt = COMMIT_REVIEW_PROMPT.format(commits=numbered, summary=summary)
            try:
                resp = await provider.text_chat(
                    prompt=prompt, system_prompt=EXTRACT_SYSTEM, persist=False
                )
                raw = getattr(resp, "completion_text", "") or ""
                m = re.search(r"\{.*\}", raw, flags=re.S)
                if not m:
                    continue
                payload = json.loads(m.group(0))
                close_ids = payload.get("close") or []
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[user_memory] 承诺复查失败: {e}")
                continue
            targets = []
            for idx in close_ids:
                try:
                    i = int(idx)
                except (TypeError, ValueError):
                    continue
                if 1 <= i <= len(items):
                    targets.append(items[i - 1].get("text", ""))
            if targets:
                total += self._close_commitments(uid, targets[:5])
        if total:
            logger.info(f"[user_memory] 承诺复查：关掉 {total} 条")
        return total

    def _recent_summaries(self, uid: str) -> str:
        """取最近两天的私聊纪要，供承诺复查参考。"""
        if not CORE_DB.exists():
            return ""
        try:
            con = sqlite3.connect(f"file:{CORE_DB}?mode=ro", uri=True)
            rows = con.execute(
                "SELECT day, content FROM daily WHERE session LIKE ? "
                "ORDER BY day DESC LIMIT 2",
                (f"%FriendMessage:{uid}",),
            ).fetchall()
            if not rows:
                rows = con.execute(
                    "SELECT day, content FROM daily WHERE session LIKE '%FriendMessage%' "
                    "ORDER BY day DESC LIMIT 2"
                ).fetchall()
            con.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[user_memory] 读纪要失败: {e}")
            return ""
        return "\n".join(f"{d}：{c}" for d, c in rows)

    # ---------- 关系推断 ----------

    def _group_messages_by_user(self, min_msgs: int) -> dict[str, list[str]]:
        """从 AstrBot 的群消息历史里按人聚合发言。

        用 platform_message_history 而不是分层记忆的原文，是因为原文只在
        她回复过的那一轮才记录，覆盖不到没接话的人。

        Args:
            min_msgs: 至少多少条才纳入。

        Returns:
            {QQ号: [发言, ...]}，只包含够条数的人。
        """
        if not HISTORY_DB.exists():
            return {}
        try:
            con = sqlite3.connect(f"file:{HISTORY_DB}?mode=ro", uri=True)
            rows = con.execute(
                "SELECT sender_id, content FROM platform_message_history "
                "WHERE user_id LIKE '%GroupMessage%' AND sender_name != 'bot' "
                "AND created_at > datetime('now', '-14 day') ORDER BY id ASC"
            ).fetchall()
            con.close()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[user_memory] 读群消息历史失败: {e}")
            return {}
        buckets: dict[str, list[str]] = {}
        for who, content in rows:
            if not who or not content:
                continue
            text = str(content)
            # 历史里存的是 JSON 结构，简单取 message 文本，取不到就跳过
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                    parts = payload.get("message") or []
                    text = "".join(
                        str(p.get("text") or p.get("data") or "")
                        for p in parts if isinstance(p, dict)
                    )
                except Exception:  # noqa: BLE001
                    continue
            text = text.strip()[:120]
            if not text:
                continue
            buckets.setdefault(str(who), []).append(text)
        return {k: v[-40:] for k, v in buckets.items() if len(v) >= min_msgs}

    async def _build_relations(self) -> int:
        """给群友推一句话的关系描述，写进档案的 relation 字段。

        Returns:
            实际更新的档案数。
        """
        if not self.relation_enabled:
            return 0
        provider = await self._internal_provider()
        if not provider:
            return 0
        buckets = self._group_messages_by_user(self.relation_min_msgs)
        if not buckets:
            return 0

        updated = 0
        for uid, lines in list(buckets.items())[:RELATION_BATCH]:
            data = self._load()
            prof = data.get(uid) or {}
            nick = prof.get("alias") or prof.get("name") or uid
            try:
                resp = await provider.text_chat(
                    prompt=RELATION_PROMPT.format(nick=nick, lines="\n".join(lines)),
                    persist=False,
                )
                text = (getattr(resp, "completion_text", "") or "").strip()
                text = text.splitlines()[0].strip().strip('"“”') if text else ""
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[user_memory] 推关系失败 {uid}: {e}")
                continue
            if not text or len(text) > 60:
                continue
            async with self._io_lock:
                data = self._load()
                entry = data.get(uid) or {"name": nick, "facts": []}
                entry["relation"] = text
                entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                data[uid] = entry
                self._save(data)
            updated += 1
            logger.info(f"[user_memory] 关系 {nick}: {text}")
        return updated

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
            provider = await self._internal_provider()
            if not provider:
                return 0
            prompt = COMPACT_PROMPT.format(
                target=self.compact_target,
                existing=json.dumps(facts, ensure_ascii=False),
            )
            resp = await provider.text_chat(
                prompt=prompt, system_prompt=EXTRACT_SYSTEM, persist=False
            )
            parsed = self._parse_facts(getattr(resp, "completion_text", ""))
            if not parsed:
                return 0
            compacted = self._dedupe_facts(parsed[0])[-self.max_facts:]

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
        """每分钟醒一次：周日压缩档案，每天到点推群友关系。"""
        while True:
            try:
                await asyncio.sleep(60)
                now = time.localtime()
                today = time.strftime("%Y-%m-%d")

                if (
                    self.weekly_compact
                    and (now.tm_hour, now.tm_min) == COMPACT_AT
                    and now.tm_wday == 0
                    and self._last_compact_day != today
                ):
                    self._last_compact_day = today
                    for uid in list(self._load().keys()):
                        await self._compact(uid)

                if (
                    (now.tm_hour, now.tm_min) >= COMMIT_REVIEW_AT
                    and self._last_commit_day != today
                ):
                    self._last_commit_day = today
                    await self._review_commitments()

                if (
                    self.relation_enabled
                    and (now.tm_hour, now.tm_min) >= RELATION_AT
                    and self._last_relation_day != today
                ):
                    self._last_relation_day = today
                    n = await self._build_relations()
                    if n:
                        logger.info(f"[user_memory] 本次更新了 {n} 个人的关系")
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                logger.error(f"[user_memory] 定时器异常: {e}")

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

    @filter.command("推关系")
    async def rebuild_relations(self, event: AstrMessageEvent):
        """立刻根据群聊重新推断群友关系（仅号主）"""
        if str(event.get_sender_id() or "") != OWNER_UIN:
            yield event.plain_result("只有号主能触发")
            return
        n = await self._build_relations()
        data = self._load()
        lines = []
        for uid, v in data.items():
            rel = (v.get("relation") or "").strip()
            if rel:
                lines.append(f"{v.get('alias') or v.get('name') or uid}：{rel}")
        if not lines:
            yield event.plain_result("推不出来，可能群聊样本还不够")
            return
        yield event.plain_result(f"更新了 {n} 个\n" + "\n".join(lines))

    @filter.command("别名")
    async def set_alias(self, event: AstrMessageEvent):
        """给某个号设别名：/别名 2738551305 猴子"""
        if str(event.get_sender_id() or "") != OWNER_UIN:
            yield event.plain_result("只有号主能改")
            return
        parts = (event.message_str or "").replace("/别名", "").split()
        if len(parts) < 2:
            yield event.plain_result("用法：/别名 2738551305 猴子")
            return
        uid, alias = parts[0], parts[1]
        async with self._io_lock:
            data = self._load()
            entry = data.get(uid) or {"name": uid, "facts": []}
            entry["alias"] = alias
            entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            data[uid] = entry
            self._save(data)
        yield event.plain_result(f"{uid} 的别名设成「{alias}」了")

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
