import os
from dotenv import load_dotenv
import asyncio
from typing import Annotated

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant, AssistantCallContext
from livekit.plugins import deepgram, openai, silero

load_dotenv()

async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track


async def entrypoint(ctx: JobContext):
    ctx.api_key = os.getenv("LIVEKIT_API_KEY")
    ctx.api_secret = os.getenv("LIVEKIT_API_SECRET")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    deepgram.api_key = os.getenv("DEEPGRAM_API_KEY")

    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Jack. You are a funny, witty bot. Your interface with users will be voice and vision."
                    "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o-mini")

    # Since OpenAI does not support streaming TTS, we'll use it with a StreamAdapter
    # to make it compatible with the VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    
    class AssistantFunction(agents.llm.FunctionContext):
        """This class is used to define functions that will be called by the assistant."""

        @agents.llm.ai_callable(
            description=(
                "Called when asked to evaluate something that would require vision capabilities,"
                "for example, an image, video, or the webcam feed."
            )
        )
        async def image(
            self,
            user_msg: Annotated[
                str,
                agents.llm.TypeInfo(
                    description="The user message that triggered this function"
                ),
            ],
        ):
            nonlocal latest_image
            print(f"Message triggering vision capabilities: {user_msg}")
            return latest_image


    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # We'll use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(api_key=deepgram.api_key),  # We'll use Deepgram's Speech To Text (STT)
        llm=gpt,
        tts=openai_tts,  # We'll use OpenAI's Text To Speech (TTS)
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    # chat = rtc.ChatManager(ctx.room)

    # async def _answer(text: str, use_image: bool = False):
    #     nonlocal latest_image
    #     """
    #     Answer the user's message with the given text and optionally the latest
    #     image captured from the video track.
    #     """
    #     content: list[str | ChatImage] = []
    #     print(f"Latest image in _answer: {latest_image}")  # Debug print
    #     if use_image and latest_image:
    #         content.append(ChatImage(image=latest_image))
        
    #     chat_context.messages.append(ChatMessage(role="user", content=content))

    #     stream = gpt.chat(chat_ctx=chat_context)
    #     await assistant.say(stream, allow_interruptions=True)

    # @assistant.on("function_calls_finished")
    # def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
    #     """This event triggers when an assistant's function call completes."""
    #     # print(called_functions)
    #     print("function calls finished")
    #     if len(called_functions) == 0:
    #         return
    #     if latest_image:
    #         chat_context.messages.append(ChatMessage(role="user", content=[latest_image]))
    #     # print(f"user_msg: {user_msg}")
    #     # if user_msg:
    #         # asyncio.create_task(_answer(user_msg, use_image=True))
    
    #     return None

    assistant.start(ctx.room)

    # chat.

    await asyncio.sleep(1)
    # await chat.say("Hi there! How can I help?")
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)

    async def track_video(track: rtc.Track):
        nonlocal latest_image
        video_stream = rtc.VideoStream(track)

        async for event in video_stream:
            latest_image = event.frame

        await video_stream.aclose()

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print(f"Video track subscribed: {track.sid}")
            asyncio.create_task(track_video(track))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        # host=os.getenv("LIVEKIT_URL"),
    ))
