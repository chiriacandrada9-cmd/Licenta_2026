"""
main.py - Application entry point.

Parses CLI arguments, sets up logging, loads configuration, and starts
the system tray + orchestrator. Includes failsafe shutdown mechanisms.
"""

import argparse
import atexit
import logging
import os
import signal
import sys
import time
from pathlib import Path

import config


def setup_logging(log_level: str) -> None:
    """Configure root logger with file + console handlers."""
    config.ensure_dirs()

    log_format = "%(asctime)s [%(levelname)-7s] %(name)-25s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root.addHandler(console)

    # File handler
    log_file = config.LOG_DIR / "handvoice.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root.addHandler(file_handler)

    logging.info("Logging initialised - level=%s, file=%s", log_level, log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="HandVoiceControl",
        description="Control your PC with hand gestures and voice commands.",
    )
    parser.add_argument(
        "--mode",
        choices=["hand", "voice", "combined"],
        default=None,
        help="Override the startup mode (default: from settings.json)",
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Run without the system tray icon (useful for debugging)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the webcam preview window on startup",
    )
    return parser.parse_args()


def _force_exit():
    """Last-resort exit: kill this process tree."""
    logger = logging.getLogger("main")
    logger.warning("Force-exiting process (failsafe)")
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass
    os._exit(0)


def main() -> None:
    # -- DPI Awareness (must be first, before any GUI/ctypes calls) --
    # Without this, Windows returns virtualized coordinates on scaled
    # displays (125%/150%), causing cursor offset.  Per-Monitor V2.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass  # Graceful fallback on older Windows

    args = parse_args()

    # Load configuration
    settings = config.load()

    # Apply CLI overrides
    if args.mode:
        config.set_value("app", "mode", args.mode)
    if args.no_preview:
        config.set_value("app", "show_preview", False)

    setup_logging(config.get("app", "log_level"))

    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("  Hand & Voice PC Control -- Starting")
    logger.info("=" * 60)
    logger.info("Mode: %s", config.get("app", "mode"))
    logger.info("Preview: %s", config.get("app", "show_preview"))
    logger.info("Failsafe: press Ctrl+C or close the preview window (Q) to stop")
    # Import heavy modules here (after logging is ready) so startup
    # problems are logged clearly.
    from modules.orchestrator import Orchestrator

    orchestrator = Orchestrator()

    # Register failsafe shutdown on any exit
    def _cleanup():
        logger.info("atexit: cleaning up...")
        try:
            orchestrator.stop()
        except Exception:
            pass
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    atexit.register(_cleanup)

    # Handle Ctrl+C and SIGTERM gracefully
    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info("Signal %s received -- shutting down...", sig_name)
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.no_tray:
        # Run directly -- blocking. Ctrl+C or Q to stop.
        logger.info("Running without tray (--no-tray). Press Ctrl+C to stop.")
        try:
            orchestrator.start()
            # Run the preview loop on the MAIN thread (required for cv2.imshow)
            _main_preview_loop(orchestrator, logger)
        except KeyboardInterrupt:
            logger.info("Ctrl+C received -- shutting down...")
        except Exception as exc:
            logger.error("Unexpected error: %s", exc, exc_info=True)
        finally:
            orchestrator.stop()
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

        # Failsafe: if threads are still stuck after 3 seconds, force exit
        logger.info("Waiting for threads to finish...")
        time.sleep(0.5)
        _force_exit()
    else:
        # Start orchestrator in a daemon thread, tray on main thread.
        from modules.tray import TrayApp

        tray = TrayApp(orchestrator)
        logger.info("System tray started. Use the tray menu to control the app.")
        try:
            tray.run()  # Blocks on main thread
        except Exception:
            pass
        finally:
            orchestrator.stop()
            _force_exit()

    logger.info("Application exited cleanly.")


def _main_preview_loop(orchestrator, logger):
    """
    Run the OpenCV preview window on the main thread.

    cv2.imshow + cv2.waitKey must be called from the main thread on
    many platforms for reliable window rendering.
    """
    import cv2

    show_preview = config.get("app", "show_preview")
    if not show_preview:
        # No preview -- just block until orchestrator stops
        orchestrator.wait()
        return

    logger.info("Preview window active on main thread. Press Q to quit.")

    while orchestrator.is_running():
        frame = orchestrator.get_preview_frame()
        if frame is not None:
            cv2.imshow("Hand & Voice Control - Preview", frame)

        key = cv2.waitKey(16) & 0xFF  # ~60 fps
        if key == ord("q"):
            logger.info("Q pressed -- shutting down...")
            break
        elif key == 27:  # ESC
            logger.info("ESC pressed -- shutting down...")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
