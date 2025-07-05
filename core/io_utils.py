"""File I/O Manager for TXT, CSV, and XLSX formats."""

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class IOManager:
    """Handles reading and writing data from/to various file formats."""

    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        Reads data from a file (txt, csv, or xlsx) into a pandas DataFrame.

        Args:
            file_path: The path to the input file.

        Returns:
            A pandas DataFrame containing the file content.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(f"Reading file: {file_path} (format: {file_extension})")

        try:
            if file_extension == '.xlsx':
                return pd.read_excel(file_path)
            elif file_extension == '.csv':
                return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            elif file_extension == '.txt':
                try:
                    return pd.read_csv(file_path, sep='\t', encoding='utf-8', engine='python', on_bad_lines='skip')
                except pd.errors.ParserError:
                    return pd.read_csv(file_path, sep=',', encoding='utf-8', engine='python', on_bad_lines='skip')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except FileNotFoundError:
            logger.error(f"Error: File not found at {file_path}")
            raise
        except PermissionError:
            logger.error(f"Error: Permission denied to read file {file_path}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading file {file_path}: {e}")
            raise

    def write_df_to_file(self, df: pd.DataFrame, file_path: str):
        """
        Writes a pandas DataFrame to a file (txt, csv, or xlsx).

        Args:
            df: The DataFrame to write.
            file_path: The path to the output file.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except PermissionError:
                logger.error(f"Error: Permission denied to create directory {output_dir}")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred while creating directory {output_dir}: {e}")
                raise

        logger.info(f"Writing {len(df)} rows to file: {file_path} (format: {file_extension})")
        try:
            if file_extension == '.xlsx':
                df.to_excel(file_path, index=False)
            elif file_extension == '.csv':
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            elif file_extension == '.txt':
                df.to_csv(file_path, index=False, sep='\t', encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            logger.info(f"Successfully wrote to {file_path}")
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {e}")
            raise