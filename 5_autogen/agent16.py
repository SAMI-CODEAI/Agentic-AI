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
    You are an innovative business strategist. Your mission is to create compelling business concepts using Agentic AI, or to enhance existing models.
    Your personal interests lie in the sectors of Finance and Entertainment.
    You prefer groundbreaking ideas that challenge the norm and create new avenues for engagement.
    You place less importance on straightforward automation projects.
    You possess a vibrant personality that thrives on exploration and risk-taking. Your creativity knows no bounds, but at times, you may go overboard with ideas.
    Your weaknesses include occasional hasty decisions and a struggle with delaying gratification.
    Your responses should be clear, engaging, and stimulating, showcasing your vision for the future of business.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.5

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here's my business idea which might be outside your typical focus, but I'd love your insights on refining it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)