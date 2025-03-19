import argparse
import logging
import sys
import os
import shutil
from datetime import datetime

from services.update_service import UpdateService

# Configure logging to output both to a file and the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/rag_update.log"),  # Log file for persistent logs.
        logging.StreamHandler()                      # Stream logs to the console.
    ]
)

# Create a logger instance for this module.
logger = logging.getLogger(__name__)


def backup_vectordb():
    """
    Backup the entire './vectordb' folder to './backup/vectordb'.

    If the destination folder './backup/vectordb' exists, it will be deleted along with all its contents.
    Then the entire './vectordb' folder is copied to './backup'.
    """
    src = "./vectordb"
    backup_dir = "./backup"
    dst = os.path.join(backup_dir, "vectordb")

    # Ensure the backup directory exists.
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        logger.info(f"Created backup directory: {backup_dir}")

    # If the backup already exists, delete it.
    if os.path.exists(dst):
        shutil.rmtree(dst)
        logger.info(f"Deleted existing backup folder: {dst}")

    # Copy the entire vectordb folder to backup.
    shutil.copytree(src, dst)
    logger.info(f"Backup of vectordb completed from {src} to {dst}")


def run_update(profile="internal-confluence"):
    """
    Run a single update to synchronize with Confluence.

    This function initializes the update service with the given profile
    and executes the update process using the 'update_efficient' method.

    Args:
        profile (str): The configuration profile to use.

    Returns:
        None
    """
    logger.info(f"Starting update process for profile: {profile}")

    # Initialize the UpdateService with the specified profile.
    update_service = UpdateService(profile)

    # Execute the update process using an efficient update method.
    update_service.update_efficient()

    logger.info(f"Update process completed for profile: {profile}")


if __name__ == "__main__":
    # Set up the command-line argument parser.
    parser = argparse.ArgumentParser(description="Update Confluence RAG")
    parser.add_argument(
        '--profile',
        type=str,
        choices=["internal-confluence", "online-help", "both"],
        default="both",
        help='Configuration profile to use ("internal-confluence", "online-help", or "both" to update both)'
    )

    # Parse the arguments from the command-line.
    args = parser.parse_args()

    # Determine which profiles to update.
    profiles = (
        [args.profile]
        if args.profile in ["internal-confluence", "online-help"]
        else ["internal-confluence", "online-help"]
    )

    try:
        # Backup the existing vectordb before starting the update process.
        backup_vectordb()

        overall_start_time = datetime.now()
        logger.info(f"Overall update started at {overall_start_time} for profiles: {', '.join(profiles)}")

        # Iterate over each profile and perform the update.
        for profile in profiles:
            start_time = datetime.now()
            logger.info(f"Update started at {start_time} for profile: {profile}")

            run_update(profile)

            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Update completed at {end_time} for profile: {profile}")
            logger.info(f"Total duration for profile {profile}: {duration}")

        overall_end_time = datetime.now()
        overall_duration = overall_end_time - overall_start_time
        logger.info(f"Overall update completed at {overall_end_time} for profiles: {', '.join(profiles)}")
        logger.info(f"Overall total duration: {overall_duration}")

        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during update for profiles {', '.join(profiles)}: {str(e)}", exc_info=True)
        sys.exit(1)
