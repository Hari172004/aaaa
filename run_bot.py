"""
run_bot.py — Agni-V Gold Bot | XAUUSD Professional Runner
=========================================================
Gold-only headless runner. No questions asked — boots instantly.
"""

import os
import time
import logging
import threading
from core import AgniVBot, BotConfig  # type: ignore
from logger import setup_file_logging  # type: ignore
from dotenv import load_dotenv         # type: ignore
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

if __name__ == "__main__":
    load_dotenv(override=True)

    # ── File logging (5MB x 3 rotating) ─────────────────────────────────
    setup_file_logging(log_file="agniv_bot.log", max_bytes=5*1024*1024, backup_count=3)

    # ── Strategy Selection (INTERACTIVE) ──────────────────────────────
    console = Console()
    env_strategy = os.getenv("BOT_STRATEGY", "PROMPT").upper()
    
    if env_strategy == "PROMPT":
        console.print(Panel("[bold cyan]AGNI-V GOLD BOT[/]\n[white]Select your trading strategy for this session:[/]", expand=False))
        console.print("  [bold yellow]1.[/] SCALP")
        console.print("  [bold yellow]2.[/] SWING")
        console.print("  [bold yellow]3.[/] AUTO")
        
        choice = Prompt.ask(
            "\n[bold white]Enter choice[/]",
            choices=["1", "2", "3"],
            default="1"
        )
        selected_strategy = {"1": "SCALP", "2": "SWING", "3": "AUTO"}[choice]
    else:
        selected_strategy = env_strategy if env_strategy in ["SCALP", "SWING", "AUTO"] else "SCALP"

    leverage = int(os.getenv("BOT_LEVERAGE", "1000"))

    console.print(f"\n[bold green]>>> Launching with Strategy:[/] [bold white]{selected_strategy}[/]\n")

    use_mtf_smc = os.getenv("USE_MTF_SMC", "true").lower() == "true"

    config = BotConfig(
        mode              = os.getenv("BOT_MODE", "REAL"),
        assets            = "XAUUSD",
        strategy          = selected_strategy,
        risk_pct          = float(os.getenv("BOT_RISK_PCT", "2.0")),
        leverage          = leverage,
        mt5_account       = int(os.getenv("MT5_ACCOUNT", "0") or "0"),
        mt5_password      = os.getenv("MT5_PASSWORD", ""),
        mt5_server        = os.getenv("MT5_SERVER", ""),
        firm              = os.getenv("FUNDED_FIRM", "FTMO"),
        firm_balance      = float(os.getenv("FUNDED_BALANCE", "10000")),
        use_ai_confirmation   = os.getenv("USE_AI_CONFIRMATION", "true").lower() == "true",
        sniper_mode           = os.getenv("SNIPER_MODE", "false").lower() == "true",
        micro_scalp           = (selected_strategy == "SCALP"),
        use_diy_strategy      = True,
        diy_scalp_config      = "diy_scalp_config.json",
        diy_swing_config      = "diy_swing_config.json",
        use_mtf_smc           = use_mtf_smc,
    )

    console.print("[bold cyan]\n⚙️  Initialising Agni-V engine... (this takes ~10s)[/]")
    console.print("[dim]  → Loading AI models, connecting MT5, building strategy...[/]")
    bot = AgniVBot(config)
    console.print("[bold green]✅ Engine ready![/]")

    # ── Telegram command handler ─────────────────────────────────────────
    try:
        from telegram_bot import TelegramCommandHandler  # type: ignore
        tg_handler = TelegramCommandHandler(
            token         = os.getenv("TELEGRAM_BOT_TOKEN", ""),
            owner_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").split(",")[0].strip(),
            allowed_ids   = os.getenv("TELEGRAM_ALLOWED_IDS", os.getenv("TELEGRAM_CHAT_ID", "")),
        )
        tg_handler.set_bot(bot)
        tg_handler.start()
        bot.alerts.telegram_chat_id = ",".join(tg_handler.subscribers)
        tg_enabled = True
    except Exception as e:
        logging.warning(f"[Startup] Telegram handler skipped: {e}")
        tg_handler = None
        tg_enabled = False

    # ── Start bot main loop ──────────────────────────────────────────────
    console.print("[bold cyan]🚀 Starting trading loop...[/]")
    bot_thread = threading.Thread(target=bot.start, daemon=True, name="AgniVCore")
    bot_thread.start()
    console.print("[bold green]✅ Bot loop active.[/]")

    # ── Console 2FA Notification ─────────────────────────────────────────
    if config.mode in ("REAL", "FUNDED") and os.getenv("BYPASS_2FA", "false").lower() != "true":
        console.print("[bold yellow]⏳ Waiting for 2FA approval on Telegram...[/]")
        console.print("[dim]Please check your Telegram bot and send /authorize[/]\n")

    # ── Startup Telegram notification ────────────────────────────────────
    def _startup_notify():
        strat_label = {"SCALP": "⚡ SCALP (M5 · DIY Builder)", "SWING": "🎯 SWING (H4 · DIY Builder)"}.get(
            selected_strategy, selected_strategy
        )
        mtf_label = "15m→5m→1m [3-Gate SMC]" if use_mtf_smc else "DIY Builder"
        time.sleep(2)
        try:
            bot.alerts.send_telegram(
                f"🟢 <b>Agni-V Gold Bot ONLINE</b>\n"
                f"{'─' * 32}\n"
                f"📈 Asset:     <code>XAUUSD (Gold)</code>\n"
                f"🎯 Strategy:  <code>{strat_label}</code>\n"
                f"⚙️  Mode:      <code>{config.mode}</code>\n"
                f"🧠 Engine:    <code>{mtf_label}</code>\n"
                f"📦 Pyramid:   <code>Up to 5 orders on strong signals</code>\n"
                f"🔫 Sniper:    <code>{'ON' if config.sniper_mode else 'OFF'}</code>\n\n"
                f"Live Gold signals active! 🚀🥇"
            )
        except Exception:
            pass

    threading.Thread(target=_startup_notify, daemon=True).start()

    # ── Heartbeat every 30 minutes ───────────────────────────────────────
    _start_time = time.time()

    def _heartbeat_loop():
        time.sleep(60)
        while True:
            try:
                info       = bot._get_balance()
                bal        = float(info.get("balance", 0.0))
                risk_stats = bot.gold_risk.stats()
                pnl        = float(risk_stats.get("daily_profit", 0.0)) - float(risk_stats.get("daily_loss", 0.0))
                open_pos   = bot._get_open_positions()
                trades     = len(open_pos) if open_pos else 0
                uptime     = int((time.time() - _start_time) / 60)
                bot.alerts.send_heartbeat(
                    balance=bal, open_trades=trades,
                    today_pnl=pnl, uptime_mins=uptime,
                )
            except Exception as e:
                logging.warning(f"[Heartbeat] {e}")
            time.sleep(30 * 60)

    threading.Thread(target=_heartbeat_loop, daemon=True, name="Heartbeat").start()
    console.print("[bold green]✅ Heartbeat monitor active.[/]")
    console.print("[bold white]─────────────────────────────────────────────────[/]")
    console.print("[bold yellow]🟢 Agni-V is LIVE. Dashboard launching... Press Ctrl+C to stop.[/]")
    console.print("[bold white]─────────────────────────────────────────────────[/]")

    logging.info(f"[Startup] Agni-V Gold Bot running | Mode={config.mode} | Strategy={selected_strategy}")

    # ── Keep alive ───────────────────────────────────────────────────────
    try:
        while True:
            if tg_enabled and tg_handler and tg_handler.subscribers:
                bot.alerts.telegram_chat_id = ",".join(tg_handler.subscribers)
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Stopping Agni-V Gold Bot...")
    except Exception as e:
        logging.error(f"[Main] Critical crash: {e}")
    finally:
        if tg_enabled and bot:
            try:
                bot.alerts.send_telegram(
                    f"🔴 <b>Agni-V Gold Bot OFFLINE</b>\n"
                    f"{'─' * 32}\n"
                    f"⚠️ The bot has stopped or been disconnected.\n"
                    f"🏦 Final Balance: <code>${bot._get_balance().get('balance', 0):.2f}</code>"
                )
                time.sleep(1)
            except:
                pass
        
        if tg_enabled and tg_handler:
            tg_handler.stop()
        bot.stop()
        console.print("[bold red]>>> AGNI-V OFFLINE. System exit.[/]")
