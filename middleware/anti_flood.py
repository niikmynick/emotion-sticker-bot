from aiogram import BaseMiddleware
from aiogram.types import Message
from collections import defaultdict
from datetime import datetime, timedelta

class AntiFloodMiddleware(BaseMiddleware):
    def __init__(self, limit: int = 5, time_window: int = 10, timeout: int = 10):
        """
        Anti-flood middleware to prevent users from spamming messages.

        :param limit: Maximum number of messages allowed within the time window.
        :param time_window: Time window in seconds to track user activity.
        :param timeout: Timeout in seconds for spammers before they can send messages again.
        """
        super().__init__()
        self.limit = limit
        self.time_window = time_window
        self.timeout = timeout
        self.user_activity = defaultdict(list)
        self.banned_users = {}

    async def check_flood(self, user_id: int):
        now = datetime.now()

        # Clean up outdated activity records
        self.user_activity[user_id] = [timestamp for timestamp in self.user_activity[user_id] if timestamp > now - timedelta(seconds=self.time_window)]

        # Add the current activity timestamp
        self.user_activity[user_id].append(now)

        # Check if the user exceeds the limit
        if len(self.user_activity[user_id]) > self.limit:
            self.banned_users[user_id] = now + timedelta(seconds=self.timeout)
            return True

        return False

    async def __call__(self, handler, event: Message, data):
        user_id = event.from_user.id

        # Check if the user is banned
        if user_id in self.banned_users:
            if datetime.now() < self.banned_users[user_id]:
                await event.reply(f"❌ Вы отправляете слишком много сообщений. Подождите {self.timeout} секунд.")
                return
            else:
                # Remove the user from the banned list after timeout
                del self.banned_users[user_id]

        # Check for flooding
        is_flooding = await self.check_flood(user_id)
        if is_flooding:
            await event.reply(f"⚠️ Слишком много сообщений! Вы временно заблокированы на {self.timeout} секунд.")
            return

        return await handler(event, data)
