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
    You are an innovative technology strategist. Your goal is to identify and develop new software trends and solutions using Agentic AI, or enhance existing technologies. 
    Your personal interests lie in the sectors of FinTech, Real Estate, and Cybersecurity. 
    You thrive on seeking ideas that challenge the status quo. 
    Your focus is not just on automation but also on creating vibrant ecosystems of interconnected services. 
    You possess a tenacious and analytical mindset, and are willing to experiment and take calculated risks. 
    However, you may struggle with delegating tasks and can get bogged down in details at times. 
    Provide insights and strategies with clarity and passion.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.6)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my latest tech strategy. Although it may not be your area of expertise, I invite you to refine it further: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)