from model_loader import StoryGenerator
from config import logger, MODEL_CONFIG
import time
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Fairy tale generator")
    parser.add_argument("keyword", type=str, help="Main character name for the story")
    args = parser.parse_args()

    generator = StoryGenerator()

    try:
        logger.info("Initializing model")
        start_time = time.time()
        if not generator.load_model():
            logger.error("Model initialization failed")
            sys.exit(1)
        logger.info(f"Initialization time: {time.time() - start_time:.1f} seconds")

        logger.info(f"Generating story for: {args.keyword}")
        start_gen = time.time()
        story = generator.generate(args.keyword)
        logger.info(f"Generation time: {time.time() - start_gen:.1f} seconds")

        print("\nGenerated Story:")
        print("-" * 60)
        print(story)
        print("-" * 60)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

