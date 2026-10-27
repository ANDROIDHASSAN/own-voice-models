"""Voice agent on open-source models only: LiveKit orchestrates, everything else runs on this machine's CPU.

    llama-server -hf ggml-org/gemma-3-1b-it-qat-GGUF --port 8080 -c 4096    # the LLM, in another terminal
    python agent.py console                                                 # talk to it, no LiveKit server needed
"""

import logging
import os

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, MetricsCollectedEvent, TurnHandlingOptions, cli, metrics
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from speech import LocalSTT, LocalTTS

LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8080/v1")  # any OpenAI-compatible server; llama-server by default
INSTRUCTIONS = """You are a friendly voice assistant talking to someone out loud.
Reply in the language the user just spoke: English, Hindi (in Devanagari) or Marathi (in Devanagari).
Keep every reply under 30 words, in plain spoken sentences. No lists, no markdown, no emoji.
If you do not know something, say so."""
logger = logging.getLogger("own-voice")


def prewarm(proc: JobProcess) -> None:
    # once per worker process, not once per call
    proc.userdata.update(vad=silero.VAD.load(), stt=LocalSTT(), tts=LocalTTS())


server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session(agent_name="own-voice")
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=ctx.proc.userdata["stt"],
        llm=openai.LLM(model="local", base_url=LLM_URL, api_key="none", temperature=0.4, max_completion_tokens=120),
        tts=ctx.proc.userdata["tts"],
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),  # LiveKit's open end-of-turn model, also on CPU
            interruption={"resume_false_interruption": True, "false_interruption_timeout": 1.0},  # a cough doesn't cut the reply
        ),
        tts_text_transforms=["filter_emoji", "filter_markdown"],
    )

    @session.on("metrics_collected")
    def _log_latency(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics, logger=logger)  # per-stage timings for every turn

    await session.start(agent=Agent(instructions=INSTRUCTIONS), room=ctx.room)
    session.generate_reply(instructions="Greet the user in one short sentence in English and ask how you can help.")


if __name__ == "__main__":
    cli.run_app(server)
