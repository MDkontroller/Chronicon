# notifier.py
import os, smtplib, ssl
from email.mime.text import MIMEText
from email.utils import formataddr

def _mask(s, keep=2):
    if not s:
        return ""
    return s[:keep] + "…" if len(s) > keep else "…"

def _mk_msg(subject, html_body, from_addr, to_addr):
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    return msg

def _send_ssl(host, port, user, pwd, envelope_from, to_addr, msg, timeout=30):
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=ctx, timeout=timeout) as s:
        s.login(user, pwd)
        s.sendmail(envelope_from, [to_addr], msg.as_string())

def _send_starttls(host, port, user, pwd, envelope_from, to_addr, msg, timeout=30):
    with smtplib.SMTP(host, port, timeout=timeout) as s:
        s.ehlo()
        s.starttls(context=ssl.create_default_context())
        s.ehlo()
        s.login(user, pwd)
        s.sendmail(envelope_from, [to_addr], msg.as_string())

def send_email(subject: str, html_body: str) -> str:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port_env = os.getenv("SMTP_PORT", "465").strip()
    try:
        port = int(port_env)
    except:
        port = 465

    user = os.getenv("SMTP_USER")         # your full Gmail address
    pwd  = os.getenv("SMTP_PASSWORD")     # the 16-char App Password
    to   = os.getenv("SMTP_TO", user)
    from_display = os.getenv("SMTP_FROM", f"Chronicon Agent <{user}>")

    if not user or not pwd:
        raise RuntimeError("SMTP_USER / SMTP_PASSWORD missing. Use your Gmail and a 16-char App Password.")

    # Prepare addresses
    if "<" in from_display and ">" in from_display:
        from_header = from_display
        envelope_from = user  # envelope sender must be the real account
    else:
        from_header = formataddr(("Chronicon Agent", user))
        envelope_from = user

    msg = _mk_msg(subject, html_body, from_header, to)

    # Try the configured port first, then auto-fallback if needed
    try:
        if port == 465:
            _send_ssl(host, 465, user, pwd, envelope_from, to, msg)
            return f"sent via SSL:{host}:465"
        elif port == 587:
            _send_starttls(host, 587, user, pwd, envelope_from, to, msg)
            return f"sent via STARTTLS:{host}:587"
        else:
            raise RuntimeError(f"Unsupported SMTP_PORT {port}. Use 465 (SSL) or 587 (STARTTLS).")
    except smtplib.SMTPAuthenticationError as e:
        # Most common: not using an App Password, or wrong account
        detail = e.smtp_error.decode('utf-8', 'ignore') if isinstance(e.smtp_error, bytes) else str(e.smtp_error)
        raise RuntimeError(
            f"SMTP auth failed ({e.smtp_code}): {detail}. "
            "Ensure 2FA is ON and you're using a Gmail App Password."
        ) from e
    except smtplib.SMTPServerDisconnected as e:
        # Auto-fallback between 465 and 587
        if port == 465:
            try:
                _send_starttls(host, 587, user, pwd, envelope_from, to, msg)
                return f"sent via STARTTLS fallback:{host}:587"
            except Exception as e2:
                raise RuntimeError(f"Server disconnected on 465 and 587 failed too: {e2}") from e
        elif port == 587:
            try:
                _send_ssl(host, 465, user, pwd, envelope_from, to, msg)
                return f"sent via SSL fallback:{host}:465"
            except Exception as e2:
                raise RuntimeError(f"Server disconnected on 587 and 465 failed too: {e2}") from e
        else:
            raise RuntimeError(f"Server disconnected on port {port}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"SMTP send error ({host}:{port} as {_mask(user)}…): {e}") from e
