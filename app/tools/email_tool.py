import asyncio
from ..logger_config import logger


async def generate_email(to: str, subject: str, tone: str = "formal"):
    logger.info("generate_email called")

    try:
        # simulate delay
        await asyncio.sleep(1)

        email_content = f"""
To: {to}
Subject: {subject}

Hello {to},

This is a {tone} email about "{subject}".

Regards,
AI Assistant
"""

        logger.info("Email generated successfully")
        logger.debug(f"Generated email content: {email_content.strip()}")

        return email_content

    except Exception as e:
        logger.error("Error while generating email", exc_info=True)
        return {"error": str(e)}