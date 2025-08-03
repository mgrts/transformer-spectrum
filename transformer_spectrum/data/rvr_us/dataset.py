from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.preprocessing import StandardScaler

from transformer_spectrum.config import (EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, SEQUENCE_LENGTH)
from transformer_spectrum.data.utils import slice_array_to_chunks

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / 'rvr_us_dataset.csv',
    output_path: Path = PROCESSED_DATA_DIR / 'rvr_us_dataset.npy',
    # time_series: str = 'total_admissions_all_influenza_confirmed_past_7days',
    time_series: str = 'average_inpatient_beds_occupied',
    rolling_window: int = 5,
    sequence_length: int = SEQUENCE_LENGTH,
):
    logger.info('Processing RVR US Hospitalization data')

    rvr_data = pd.read_csv(input_path)

    # Sort data by jurisdiction and date
    rvr_data['collection_date'] = pd.to_datetime(rvr_data['collection_date'])
    rvr_data = rvr_data.sort_values(['jurisdiction', 'collection_date'])

    # Handle missing values by forward-filling within each jurisdiction
    rvr_data[time_series] = (
        rvr_data.groupby('jurisdiction')[time_series]
        .transform(lambda x: x.ffill().bfill())
    )

    # Apply smoothing (5-day rolling average)
    rvr_data[f'smoothed_{time_series}'] = (
        rvr_data.groupby('jurisdiction')[time_series]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
    )

    sequences = []
    jurisdictions = rvr_data['jurisdiction'].unique()

    for jurisdiction in jurisdictions:
        ts_data = rvr_data[rvr_data['jurisdiction'] == jurisdiction][f'smoothed_{time_series}'].values
        chunks = slice_array_to_chunks(ts_data, sequence_length)

        for chunk in chunks:
            chunk_scaled = StandardScaler().fit_transform(chunk.reshape(-1, 1)).reshape(-1)
            sequences.append(chunk_scaled)

    sequences = np.vstack(sequences)
    sequences = sequences[..., np.newaxis]

    logger.info('Saving processed data')

    with open(output_path, 'wb') as f:
        np.save(f, sequences)

    logger.success('Processing RVR dataset complete.')


if __name__ == '__main__':
    app()
