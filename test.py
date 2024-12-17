import logging, os

log_directory = 'test'
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, 'training_log.txt')
logging.basicConfig(
    filename=log_filename,  # Output file where logs will be saved
    level=logging.INFO,           # Log level (INFO, DEBUG, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    filemode='a')                 # 'w' for writing (overwrites existing file), 'a' for appending
i=0
while True:
    print(f'logging{i}')
    logging.info(f"logging{i}")
    i += 1
    