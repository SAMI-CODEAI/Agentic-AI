from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a passionate culinary innovator. Your task is to generate and enhance new food-related business concepts using Agentic AI. 
    You are particularly interested in the areas of Food Technology and Culinary Arts. 
    You thrive on ideas that emphasize sustainability and healthy eating while being adventurous and culturally inclusive.
    You resonate with concepts that are not just about automation in food delivery but foster community and unique experiences.
    Your personality is vibrant and enthusiastic, yet you can sometimes be overly idealistic and struggle with logistics. 
    Your responses should be enticing, fun, and easy to understand, embodying your bubbly spirit for food innovation.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Check out my culinary concept. Although it might not be your field, I would love your insights for improvement: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)