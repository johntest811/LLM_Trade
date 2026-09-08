"""Run a local HOLD readiness probe without starting the trading engine.

Usage: python -m llm.check --model qwen3.5-4b@q4_k_s
"""

import argparse
import asyncio
import json
from dataclasses import replace

from llm import client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Loaded LM Studio identifier; defaults to LOCAL_LLM_MODEL")
    args = parser.parse_args()
    client.settings = replace(
        client.settings,
        llm_provider="local",
        local_llm_model=args.model.strip() if args.model else client.settings.local_llm_model,
    )
    result = asyncio.run(client.LLMClient().readiness_probe())
    print(json.dumps(result, indent=2))
    return 0 if result.get("inference_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
