from pathlib import Path

import requests
import typer
from loguru import logger
from tqdm import tqdm

from transformer_spectrum.config import DATA_URL, EXTERNAL_DATA_DIR

app = typer.Typer(pretty_exceptions_show_locals=False)


def download_file_with_progress(url: str, dest_path: Path, chunk_size: int = 1024) -> Path:
    """
    Downloads a file from a URL with a progress bar.

    Args:
        url (str): URL of the file to download.
        dest_path (Path): Local path to save the downloaded file.
        chunk_size (int): Size of each chunk to read. Defaults to 1024 bytes.

    Returns:
        Path: The path to the saved file.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f'Failed to start download: {e}')
        raise typer.Exit(code=1)

    total_size = int(response.headers.get('content-length', 0))

    with tqdm(total=total_size, unit='iB', unit_scale=True, desc='Downloading') as progress_bar, \
            open(dest_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

    if total_size != 0 and progress_bar.n != total_size:
        logger.error('Download incomplete or corrupted.')
        raise typer.Exit(code=1)

    return dest_path


@app.command()
def download(
        input_url: str = typer.Option(DATA_URL, help='URL to download the dataset from.'),
        output_path: Path = typer.Option(EXTERNAL_DATA_DIR / 'dataset.csv', help='Path to save the downloaded dataset.'),
):
    logger.info(f'Starting download from {input_url}')

    final_path = download_file_with_progress(input_url, output_path)

    logger.success(f'Download complete: {final_path}')


if __name__ == '__main__':
    app()
