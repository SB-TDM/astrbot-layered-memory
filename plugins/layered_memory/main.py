"""Layered memory (7 layers, implemented as 6 stores + per-turn scratch).

L1 core    — who you are / who this person is / the relationship. Always injected.
L2 monthly — one summary per month, built from weeklies.
L3 weekly  — one summary per week, built from dailies.
L4 daily   — one summary per day, built at 23:50 from raw messages.
L5 recent  — rolling summary of the last N messages, refreshed every N/2 turns.
L6 raw     — original messages, never injected, deleted after RAW_KEEP_DAYS.
L7 scratch — the current turn's working notes; not persisted.

Commands:
    /记忆层        show L1..L5
    /记住 <内容>   append a line to your L1 core (owner only)
    /总结          build today's daily summary now
    /周总结        build this week's summary now
    /月总结        build this month's summary now
"""

import asyncio
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

DB_FILE = Path(__file__).resolve().parent / "memories.db"

OWNER_UIN = "YOUR_QQ_NUMBER"
RAW_KEEP_DAYS = 3
DAILY_INJECT = 2
RECENT_TURNS = 20          # L5 每收到这么多轮就刷新一次
CORE_MAX_CHARS = 500
DAILY_AT = (23, 50)
WEEKLY_AT = (23, 55)       # 只在周日触发
MONTHLY_AT = (0, 5)        # 只在每月 1 号触发
CHECK_EVERY_SEC = 60

SUMMARY_SYSTEM = (
    "你是一个记忆整理器。只输出整理后的内容，不要解释，不要加标题。"
    "语言用中文，口语一点，像给自己记的备忘。"
)

DAILY_PROMPT = """下面是今天（{day}）我和「他」的聊天记录，请压缩成 3 到 5 句的当日纪要。

要求：
- 只留事实和情绪基调：聊了什么、他状态如何、有没有约定或没说完的事。
- 不要感慨，不要复述原话，不要写「今天聊了很多」这种空话。
- 直接给纪要正文，不要加日期和标题。

聊天记录：
{chat}
"""

WEEKLY_PROMPT = """下面是这一周（{start} 到 {end}）的每日纪要，请压缩成 4 到 6 句的周纪要。

要求：
- 合并重复的事，只留主线：这周主要聊了什么、他的状态起伏、有什么没解决的。
- 不要逐日复述，不要加「这周」以外的标题。
- 直接给纪要正文。

每日纪要：
{chat}
"""

MONTHLY_PROMPT = """下面是这个月（{month}）的每周纪要，请压缩成 5 到 7 句的月纪要。

要求：
- 只留长期有意义的：关系变化、反复出现的话题、重要决定或约定。
- 丢掉一次性小事。
- 直接给纪要正文。

每周纪要：
{chat}
"""

RECENT_PROMPT = """下面是最近一段聊天，请压成一两句的即时上下文摘要，供下一轮对话参考。

要求：
- 只写「现在在聊什么、他的情绪如何、有没有没答完的问题」。
- 不超过两句。
- 直接给内容。

聊天：
{chat}
"""


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def week_key(d: datetime | None = None) -> str:
    d = d or datetime.now()
    return f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"


def month_key(d: datetime | None = None) -> str:
    return (d or datetime.now()).strftime("%Y-%m")


class LayeredMemory(Star):
    """七层记忆：内核 + 月/周/日纪要 + 近期摘要 + 原文缓存。"""

    def __init__(self, context: Context, config=None) -> None:
        super().__init__(context)
        self.cfg = config
        self.enable = self._cfg("enable", True, bool)
        self.raw_keep_days = self._cfg("raw_keep_days", RAW_KEEP_DAYS, int)
        self.daily_inject = self._cfg("daily_inject", DAILY_INJECT, int)
        self.recent_turns = self._cfg("recent_turns", RECENT_TURNS, int)
        self.core_max_chars = self._cfg("core_max_chars", CORE_MAX_CHARS, int)
        self.daily_at = self._parse_time(self._cfg("daily_time", "23:50", str), DAILY_AT)
        self.weekly_at = self._parse_time(self._cfg("weekly_time", "23:55", str), WEEKLY_AT)
        self.monthly_at = self._parse_time(self._cfg("monthly_time", "00:05", str), MONTHLY_AT)
        self.backfill_enabled = self._cfg("backfill_enabled", True, bool)
        self.cross_session_recent = self._cfg("cross_session_recent", True, bool)
        self.cross_fresh_minutes = self._cfg("cross_fresh_minutes", 120, int)
        self.cross_in_group = self._cfg("cross_in_group", False, bool)
        self.recent_trigger = self._cfg("recent_trigger", "turns", str)
        self.recent_interval = self._cfg("recent_interval_minutes", 5, int)
        self.recent_min_msgs = self._cfg("recent_min_messages", 5, int)
        self._init_db()
        self._task: asyncio.Task | None = None
        self._done: dict[str, str] = {}      # 防止同一时间点重复触发
        self._turn_count: dict[str, int] = {}

    def _cfg(self, key: str, default, cast):
        """读插件配置项，读不到就用默认值。"""
        if not self.cfg:
            return default
        try:
            value = self.cfg.get(key, default)
            return default if value is None else cast(value)
        except Exception:  # noqa: BLE001
            return default

    @staticmethod
    def _parse_time(text: str, fallback: tuple[int, int]) -> tuple[int, int]:
        """把 "HH:MM" 解析成 (时, 分)，格式不对就用默认值。"""
        try:
            hh, mm = str(text).split(":")
            hh, mm = int(hh), int(mm)
            if 0 <= hh < 24 and 0 <= mm < 60:
                return (hh, mm)
        except Exception:  # noqa: BLE001
            pass
        return fallback

    # ---------- storage ----------

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(DB_FILE)
        con.execute(
            "CREATE TABLE IF NOT EXISTS core ("
            " key TEXT PRIMARY KEY, content TEXT, updated_at REAL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS daily ("
            " session TEXT, day TEXT, content TEXT, updated_at REAL,"
            " PRIMARY KEY (session, day))"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS weekly ("
            " session TEXT, week TEXT, content TEXT, updated_at REAL,"
            " PRIMARY KEY (session, week))"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS monthly ("
            " session TEXT, month TEXT, content TEXT, updated_at REAL,"
            " PRIMARY KEY (session, month))"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS recent ("
            " session TEXT PRIMARY KEY, content TEXT, updated_at REAL)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS raw ("
            " session TEXT, who TEXT, role TEXT, content TEXT, ts REAL)"
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw(ts)")
        return con

    def _init_db(self) -> None:
        with self._conn():
            pass

    # ---------- injection ----------

    @filter.on_llm_request()
    async def inject_layers(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """把 L1..L5 按预算拼进 system prompt。"""
        if not self.enable:
            return
        try:
            session = event.unified_msg_origin
            uid = str(event.get_sender_id() or "")
            if not session or not uid:
                return

            con = self._conn()
            core = con.execute("SELECT content FROM core WHERE key=?", (uid,)).fetchone()
            month = con.execute(
                "SELECT month, content FROM monthly WHERE session=? "
                "ORDER BY month DESC LIMIT 1", (session,)
            ).fetchone()
            week = con.execute(
                "SELECT week, content FROM weekly WHERE session=? "
                "ORDER BY week DESC LIMIT 1", (session,)
            ).fetchone()
            dailies = con.execute(
                "SELECT day, content FROM daily WHERE session=? "
                "ORDER BY day DESC LIMIT ?", (session, self.daily_inject)
            ).fetchall()
            recent = con.execute(
                "SELECT content FROM recent WHERE session=?", (session,)
            ).fetchone()
            con.close()

            blocks: list[str] = []
            if core and core[0]:
                blocks.append("【你记得的他】\n" + core[0][:self.core_max_chars])
            if month:
                blocks.append(f"【上个月概要】\n{month[1][:300]}")
            if week:
                blocks.append(f"【这周概要】\n{week[1][:300]}")
            if dailies:
                blocks.append(
                    "【最近几天】\n"
                    + "\n".join(f"{d}：{t}" for d, t in reversed(dailies))
                )
            if recent and recent[0]:
                blocks.append("【刚才在聊】\n" + recent[0][:200])

            # 跨会话：把另一个会话的即时摘要也带上，避免换个场景就接不上话
            # 默认只在私聊里带别的会话摘要；群里不带，避免私聊内容外泄
            inject_cross = self.cross_session_recent and (
                self.cross_in_group or "GroupMessage" not in session
            )
            if inject_cross:
                con = self._conn()
                other = con.execute(
                    "SELECT content FROM recent WHERE session != ? AND updated_at > ? "
                    "ORDER BY updated_at DESC LIMIT 1",
                    (session, time.time() - self.cross_fresh_minutes * 60),
                ).fetchone()
                con.close()
                if other and other[0]:
                    blocks.append("【另一个会话里刚才在聊】\n" + other[0][:150])

            if blocks:
                req.system_prompt = (
                    (req.system_prompt or "") + "\n\n" + "\n\n".join(blocks)
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[layered_memory] 注入失败: {e}")

    # ---------- recording ----------

    @filter.on_llm_response()
    async def remember_raw(self, event: AstrMessageEvent, resp: LLMResponse) -> None:
        """存原文（L6），并按轮数滚动刷新 L5。"""
        if not self.enable:
            return
        try:
            session = event.unified_msg_origin
            uid = str(event.get_sender_id() or "")
            if not session or not uid:
                return
            user_text = (event.get_message_str() or "").strip()
            reply = (getattr(resp, "completion_text", "") or "").strip()
            if not user_text:
                return

            now = time.time()
            con = self._conn()
            con.execute(
                "INSERT INTO raw (session, who, role, content, ts) VALUES (?,?,?,?,?)",
                (session, uid, "user", user_text[:800], now),
            )
            if reply:
                con.execute(
                    "INSERT INTO raw (session, who, role, content, ts) VALUES (?,?,?,?,?)",
                    (session, uid, "bot", reply[:800], now),
                )
            con.execute("DELETE FROM raw WHERE ts < ?", (now - self.raw_keep_days * 86400,))
            con.commit()
            con.close()

            if self._should_refresh_recent(session):
                asyncio.create_task(self._refresh_recent(session))
        except Exception as e:  # noqa: BLE001
            logger.error(f"[layered_memory] 记录失败: {e}")

    # ---------- summaries ----------

    async def _call_llm(self, prompt: str) -> str:
        provider = await self.context.get_using_provider_async()
        if not provider:
            return ""
        try:
            resp = await provider.text_chat(
                prompt=prompt, system_prompt=SUMMARY_SYSTEM, persist=False
            )
            return (getattr(resp, "completion_text", "") or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[layered_memory] 模型调用失败: {e}")
            return ""

    async def _summarize_session(self, session: str, day: str | None = None) -> str:
        """L6 -> L4：把一个会话当天的原文压成日纪要。"""
        day = day or today_str()
        con = self._conn()
        exists = con.execute(
            "SELECT 1 FROM daily WHERE session=? AND day=?", (session, day)
        ).fetchone()
        con.close()
        if exists:
            return
        try:
            day_start = datetime.strptime(day, "%Y-%m-%d").timestamp()
        except ValueError:
            day_start = time.time() - 86400
        day_end = day_start + 86400
        con = self._conn()
        rows = con.execute(
            "SELECT role, content FROM raw WHERE session=? AND ts >= ? AND ts < ? ORDER BY ts ASC",
            (session, day_start, day_end),
        ).fetchall()
        con.close()
        if not rows:
            return ""
        chat = "\n".join(
            f"{'我' if role == 'bot' else '他'}：{c}" for role, c in rows
        )[-6000:]
        summary = await self._call_llm(DAILY_PROMPT.format(day=day, chat=chat))
        if not summary:
            return ""
        con = self._conn()
        con.execute(
            "INSERT OR REPLACE INTO daily (session, day, content, updated_at) VALUES (?,?,?,?)",
            (session, day, summary[:800], time.time()),
        )
        con.commit()
        con.close()
        logger.info(f"[layered_memory] 日纪要 {session} {day}")
        return summary

    async def _build_weekly(self, session: str, end: datetime | None = None) -> str:
        """L4 -> L3：最近 7 天的日纪要压成周纪要。"""
        end = end or datetime.now()
        start = end - timedelta(days=6)
        con = self._conn()
        rows = con.execute(
            "SELECT day, content FROM daily WHERE session=? AND day >= ? AND day <= ? "
            "ORDER BY day ASC",
            (session, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
        ).fetchall()
        con.close()
        if not rows:
            return ""
        wk_check = week_key(end)
        con = self._conn()
        exists = con.execute(
            "SELECT 1 FROM weekly WHERE session=? AND week=?", (session, wk_check)
        ).fetchone()
        con.close()
        if exists:
            return ""
        chat = "\n".join(f"{d}：{t}" for d, t in rows)
        summary = await self._call_llm(
            WEEKLY_PROMPT.format(
                start=start.strftime("%m-%d"), end=end.strftime("%m-%d"), chat=chat
            )
        )
        if not summary:
            return ""
        wk = week_key(end)
        con = self._conn()
        con.execute(
            "INSERT OR REPLACE INTO weekly (session, week, content, updated_at) VALUES (?,?,?,?)",
            (session, wk, summary[:800], time.time()),
        )
        con.commit()
        con.close()
        logger.info(f"[layered_memory] 周纪要 {session} {wk}")
        return summary

    async def _build_monthly(self, session: str, month: str | None = None) -> str:
        """L3 -> L2：上个月的周纪要压成月纪要。"""
        month = month or month_key()
        con = self._conn()
        rows = con.execute(
            "SELECT week, content FROM weekly WHERE session=? AND week LIKE ? ORDER BY week ASC",
            (session, month[:4] + "%"),
        ).fetchall()
        if not rows:
            rows = con.execute(
                "SELECT day, content FROM daily WHERE session=? AND day LIKE ? ORDER BY day ASC",
                (session, month + "%"),
            ).fetchall()
        con.close()
        if not rows:
            return ""
        con = self._conn()
        exists = con.execute(
            "SELECT 1 FROM monthly WHERE session=? AND month=?", (session, month)
        ).fetchone()
        con.close()
        if exists:
            return ""
        chat = "\n".join(f"{k}：{v}" for k, v in rows)
        summary = await self._call_llm(MONTHLY_PROMPT.format(month=month, chat=chat))
        if not summary:
            return ""
        con = self._conn()
        con.execute(
            "INSERT OR REPLACE INTO monthly (session, month, content, updated_at) VALUES (?,?,?,?)",
            (session, month, summary[:800], time.time()),
        )
        con.commit()
        con.close()
        logger.info(f"[layered_memory] 月纪要 {session} {month}")
        return summary

    def _should_refresh_recent(self, session: str) -> bool:
        """判断 L5 是否需要刷新。

        两种模式：
            turns  — 距上次摘要又攒够 recent_turns 条用户消息
            hybrid — 至少攒够 recent_min_messages 条，且距上次摘要超过
                     recent_interval_minutes 分钟

        Args:
            session: 会话标识。

        Returns:
            需要刷新返回 True。
        """
        try:
            con = self._conn()
            last = con.execute(
                "SELECT updated_at FROM recent WHERE session=?", (session,)
            ).fetchone()
            since = last[0] if last else 0
            n = con.execute(
                "SELECT COUNT(*) FROM raw WHERE session=? AND role='user' AND ts > ?",
                (session, since),
            ).fetchone()[0]
            con.close()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[layered_memory] 判断 L5 失败: {e}")
            return False

        if str(self.recent_trigger).lower() == "hybrid":
            if n < self.recent_min_msgs:
                return False
            return (time.time() - since) >= self.recent_interval * 60
        return n >= self.recent_turns

    async def _refresh_recent(self, session: str) -> None:
        """L5：把最近 RECENT_TURNS 条原文压成一两句即时摘要。"""
        try:
            con = self._conn()
            rows = con.execute(
                "SELECT role, content FROM raw WHERE session=? ORDER BY ts DESC LIMIT ?",
                (session, self.recent_turns * 2),
            ).fetchall()
            con.close()
            if not rows:
                return
            chat = "\n".join(
                f"{'我' if role == 'bot' else '他'}：{c}" for role, c in reversed(rows)
            )[-3000:]
            summary = await self._call_llm(RECENT_PROMPT.format(chat=chat))
            if not summary:
                return
            con = self._conn()
            con.execute(
                "INSERT OR REPLACE INTO recent (session, content, updated_at) VALUES (?,?,?)",
                (session, summary[:400], time.time()),
            )
            con.commit()
            con.close()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[layered_memory] 刷新近期摘要失败: {e}")

    # ---------- scheduler ----------

    @filter.on_astrbot_loaded()
    async def start_loop(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            logger.info("[layered_memory] 分层记忆定时器已启动")
        if self.backfill_enabled:
            asyncio.create_task(self._backfill())

    async def _backfill(self) -> None:
        """补生成遗漏的日纪要。

        只处理原文还在（raw_keep_days 天内）但没有纪要的日期；
        原文已被删除的日期无法补救。
        """
        try:
            await asyncio.sleep(10)
            cutoff = time.time() - self.raw_keep_days * 86400
            con = self._conn()
            raw_days = {
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT date(ts, 'unixepoch', 'localtime') FROM raw WHERE ts > ?",
                    (cutoff,),
                )
            }
            saved = {
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT day FROM daily WHERE day >= date('now','localtime','-%d day')"
                    % max(self.raw_keep_days, 1)
                )
            }
            con.close()
            today = today_str()
            missing = sorted(d for d in raw_days - saved if d < today)
            if not missing:
                return
            logger.info(f"[layered_memory] 补生成纪要：{missing}")
            for day in missing:
                con = self._conn()
                sessions = [
                    r[0]
                    for r in con.execute(
                        "SELECT DISTINCT session FROM raw WHERE date(ts,'unixepoch','localtime')=?",
                        (day,),
                    )
                ]
                con.close()
                for session in sessions:
                    await self._summarize_session(session, day)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[layered_memory] 补生成失败: {e}")

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(CHECK_EVERY_SEC)
                await self._tick()
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                logger.error(f"[layered_memory] 定时器异常: {e}")

    async def _tick(self) -> None:
        if not self.enable:
            return
        # hybrid 模式：消息停止后也要能到点刷新，所以顺带扫一遍
        if str(self.recent_trigger).lower() == "hybrid":
            try:
                con = self._conn()
                sessions = [r[0] for r in con.execute("SELECT DISTINCT session FROM raw")]
                con.close()
                for session in sessions:
                    if self._should_refresh_recent(session):
                        asyncio.create_task(self._refresh_recent(session))
            except Exception as e:  # noqa: BLE001
                logger.error(f"[layered_memory] hybrid 检查失败: {e}")
        now = datetime.now()
        stamp = now.strftime("%Y-%m-%d %H:%M")

        # 不用「正好等于某一分钟」判断：单次 sleep 加上一轮工作总结可能跨过整分钟，
        # 一旦错过就整天不补。改成「到点且今天没做过」——晚到多久都会补上。
        daily_due = now.replace(hour=self.daily_at[0], minute=self.daily_at[1],
                                second=0, microsecond=0)
        if now >= daily_due and self._done.get("daily") != today_str():
            self._done["daily"] = today_str()
            for s in self._sessions(86400):
                await self._summarize_session(s)

        # 周纪要：最近一个「周日 + 设定时刻」。没做完就补，跨重启也能补。
        days_back = (now.isoweekday() - 7) % 7
        weekly_due = (now - timedelta(days=days_back)).replace(
            hour=self.weekly_at[0], minute=self.weekly_at[1], second=0, microsecond=0
        )
        if now >= weekly_due and self._done.get("weekly") != week_key(weekly_due):
            self._done["weekly"] = week_key(weekly_due)
            for s in self._sessions(8 * 86400):
                await self._build_weekly(s)

        # 月纪要：本月 1 号的设定时刻，到点就算，不限于当天。
        monthly_due = now.replace(day=1, hour=self.monthly_at[0],
                                 minute=self.monthly_at[1], second=0, microsecond=0)
        if now >= monthly_due:
            prev = (monthly_due - timedelta(days=1)).strftime("%Y-%m")
            if self._done.get("monthly") != prev:
                self._done["monthly"] = prev
                for s in self._sessions(40 * 86400):
                    await self._build_monthly(s, prev)

        self._done.setdefault("last_tick", stamp)

    def _sessions(self, window_sec: float) -> list[str]:
        con = self._conn()
        rows = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT session FROM raw WHERE ts > ?",
                (time.time() - window_sec,),
            )
        ]
        con.close()
        return rows

    # ---------- commands ----------

    @filter.command("记忆层")
    async def show_layers(self, event: AstrMessageEvent):
        """查看各层内容"""
        uid = str(event.get_sender_id() or "")
        session = event.unified_msg_origin
        con = self._conn()
        core = con.execute("SELECT content FROM core WHERE key=?", (uid,)).fetchone()
        month = con.execute(
            "SELECT month, content FROM monthly WHERE session=? ORDER BY month DESC LIMIT 1",
            (session,),
        ).fetchone()
        week = con.execute(
            "SELECT week, content FROM weekly WHERE session=? ORDER BY week DESC LIMIT 1",
            (session,),
        ).fetchone()
        days = con.execute(
            "SELECT day, content FROM daily WHERE session=? ORDER BY day DESC LIMIT 3",
            (session,),
        ).fetchall()
        recent = con.execute(
            "SELECT content FROM recent WHERE session=?", (session,)
        ).fetchone()
        con.close()

        out = ["【L1 内核】" + (("\n" + core[0][:250]) if core else "（空）")]
        out.append("【L2 月】" + (f"{month[0]}：{month[1][:120]}" if month else "（空）"))
        out.append("【L3 周】" + (f"{week[0]}：{week[1][:120]}" if week else "（空）"))
        if days:
            out.append("【L4 日】")
            out += [f"{d}：{t[:100]}" for d, t in days]
        else:
            out.append("【L4 日】（空）")
        out.append("【L5 即时】" + (recent[0][:120] if recent else "（空）"))
        yield event.plain_result("\n".join(out))

    @filter.command("记住")
    async def remember(self, event: AstrMessageEvent):
        """往 L1 追加：/记住 他叫小李"""
        if str(event.get_sender_id() or "") != OWNER_UIN:
            yield event.plain_result("只有号主能改内核")
            return
        text = (event.message_str or "").replace("/记住", "").strip()
        if not text:
            yield event.plain_result("要记什么？/记住 内容")
            return
        uid = str(event.get_sender_id() or "")
        con = self._conn()
        row = con.execute("SELECT content FROM core WHERE key=?", (uid,)).fetchone()
        content = ((row[0] + "\n" + text) if row and row[0] else text)[-self.core_max_chars:]
        con.execute(
            "INSERT OR REPLACE INTO core (key, content, updated_at) VALUES (?,?,?)",
            (uid, content, time.time()),
        )
        con.commit()
        con.close()
        yield event.plain_result("记下了")

    @filter.command("总结")
    async def summarize_now(self, event: AstrMessageEvent):
        """立刻生成今天的日纪要"""
        if str(event.get_sender_id() or "") != OWNER_UIN:
            yield event.plain_result("只有号主能用")
            return
        s = await self._summarize_session(event.unified_msg_origin)
        yield event.plain_result(("今天的纪要：\n" + s) if s else "今天还没聊什么")

    @filter.command("周总结")
    async def weekly_now(self, event: AstrMessageEvent):
        """立刻生成本周周纪要"""
        if str(event.get_sender_id() or "") != OWNER_UIN:
            yield event.plain_result("只有号主能用")
            return
        s = await self._build_weekly(event.unified_msg_origin)
        yield event.plain_result(("这周的纪要：\n" + s) if s else "还没有足够的日纪要")

    @filter.command("月总结")
    async def monthly_now(self, event: AstrMessageEvent):
        """立刻生成本月月纪要"""
        if str(event.get_sender_id() or "") != OWNER_UIN:
            yield event.plain_result("只有号主能用")
            return
        s = await self._build_monthly(event.unified_msg_origin)
        yield event.plain_result(("这个月的纪要：\n" + s) if s else "还没有足够的周纪要")
