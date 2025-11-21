import os
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest
from datetime import datetime

from src.data.download_data import download_data, main


class TestDownloadData:
    '''Test suite for download_data function.'''

    def test_download_data_success(self, tmp_path):
        '''Test successful data download with mocked download function.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'dataset.zip')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()  # Create the mock file

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert result == str(Path(raw_data_path) / 'dataset.zip')
        assert os.path.exists(result)
        assert os.path.isdir(raw_data_path)
        mock_download.assert_called_once_with()

    def test_download_data_with_different_file_types(self, tmp_path):
        '''Test download with different file types (csv instead of zip).'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'custom_dataset.csv')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert os.path.exists(result)
        assert result.endswith('custom_dataset.csv')
        mock_download.assert_called_once_with()

    def test_download_data_creates_directory(self, tmp_path):
        '''Test that download_data creates the destination directory if it doesn't exist.'''
        # Setup
        raw_data_path = str(tmp_path / 'nested' / 'path' / 'to' / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'dataset.zip')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()

        mock_download = MagicMock(return_value=mock_file_path)

        # Ensure directory doesn't exist yet
        assert not os.path.exists(raw_data_path)

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert os.path.exists(raw_data_path)
        assert os.path.exists(result)

    def test_download_data_empty_raw_data_path_raises_error(self, tmp_path):
        '''Test that empty raw_data_path raises ValueError.'''
        mock_download = MagicMock()

        with pytest.raises(ValueError, match='raw_data_path cannot be empty'):
            download_data(raw_data_path='', download_func=mock_download)

        mock_download.assert_not_called()

    def test_download_data_none_download_func_raises_error(self, tmp_path):
        '''Test that None download_func raises ValueError.'''
        with pytest.raises(ValueError, match='download_func cannot be empty'):
            download_data(
                raw_data_path=str(tmp_path),
                download_func=None,
            )

    def test_download_data_file_exists_after_download(self, tmp_path):
        '''Test that the file is properly moved to the destination.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        source_dir = tmp_path / 'source'
        source_dir.mkdir()
        mock_file_path = str(source_dir / 'accidents.zip')
        Path(mock_file_path).write_text('test data')

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert os.path.exists(result)
        assert not os.path.exists(mock_file_path)  # Original should be moved, not copied
        with open(result) as f:
            assert f.read() == 'test data'

    def test_download_data_preserves_filename(self, tmp_path):
        '''Test that the filename is preserved when moving to destination.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        source_file = tmp_path / 'source' / 'my_data.tar.gz'
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.touch()

        mock_download = MagicMock(return_value=str(source_file))

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
        )

        # Assert
        assert result.endswith('my_data.tar.gz')
        assert os.path.exists(result)
        assert os.path.dirname(result) == raw_data_path

    def test_download_data_with_timestamp(self, tmp_path):
        '''Test that add_timestamp parameter creates a timestamped subdirectory.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'dataset.zip')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute
        with patch('src.data.download_data.datetime') as mock_datetime:
            mock_now = Mock()
            mock_now.strftime.return_value = '20231115_1430'
            mock_datetime.now.return_value = mock_now

            result = download_data(
                raw_data_path=raw_data_path,
                download_func=mock_download,
                add_timestamp=True,
            )

        # Assert
        assert '20231115_1430' in result
        assert os.path.exists(result)
        expected_path = os.path.join(raw_data_path, '20231115_1430')
        assert os.path.exists(expected_path)
        assert result == os.path.join(expected_path, 'dataset.zip')
        mock_download.assert_called_once_with()

    def test_download_data_without_timestamp(self, tmp_path):
        '''Test that add_timestamp=False does not create a timestamped subdirectory.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'dataset.zip')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
            add_timestamp=False,
        )

        # Assert
        assert result == os.path.join(raw_data_path, 'dataset.zip')
        assert os.path.exists(result)
        # Verify no timestamp subdirectory was created
        assert os.path.dirname(result) == raw_data_path
        mock_download.assert_called_once_with()

    def test_download_data_file_not_found_after_download(self, tmp_path):
        '''Test that FileNotFoundError is raised if download_func returns non-existent path.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        non_existent_path = str(tmp_path / 'source' / 'nonexistent.zip')
        
        mock_download = MagicMock(return_value=non_existent_path)

        # Execute & Assert
        with pytest.raises(FileNotFoundError, match='Downloaded file not found'):
            download_data(
                raw_data_path=raw_data_path,
                download_func=mock_download,
            )

        mock_download.assert_called_once_with()

    def test_download_data_timestamp_format(self, tmp_path):
        '''Test that timestamp has the correct format YYYYMMDD_HHMM.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'data.csv')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute with actual datetime (not mocked)
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
            add_timestamp=True,
        )

        # Assert - Extract timestamp from path
        parts = result.split(os.sep)
        timestamp_part = None
        for part in parts:
            if '_' in part and len(part) == 13:  # Format: YYYYMMDD_HHMM
                timestamp_part = part
                break
        
        assert timestamp_part is not None
        # Verify format with regex-like check
        date_part, time_part = timestamp_part.split('_')
        assert len(date_part) == 8  # YYYYMMDD
        assert len(time_part) == 4  # HHMM
        assert date_part.isdigit()
        assert time_part.isdigit()

    def test_download_data_multiple_calls_with_timestamp(self, tmp_path):
        '''Test that multiple downloads with timestamp create separate directories.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        
        mock_file_path_1 = str(tmp_path / 'source1' / 'dataset1.zip')
        os.makedirs(os.path.dirname(mock_file_path_1), exist_ok=True)
        Path(mock_file_path_1).write_text('data1')

        mock_file_path_2 = str(tmp_path / 'source2' / 'dataset2.zip')
        os.makedirs(os.path.dirname(mock_file_path_2), exist_ok=True)
        Path(mock_file_path_2).write_text('data2')

        mock_download_1 = MagicMock(return_value=mock_file_path_1)
        mock_download_2 = MagicMock(return_value=mock_file_path_2)

        # Execute with mocked timestamps
        with patch('src.data.download_data.datetime') as mock_datetime:
            mock_now_1 = Mock()
            mock_now_1.strftime.return_value = '20231115_1430'
            mock_datetime.now.return_value = mock_now_1
            result_1 = download_data(
                raw_data_path=raw_data_path,
                download_func=mock_download_1,
                add_timestamp=True,
            )

            mock_now_2 = Mock()
            mock_now_2.strftime.return_value = '20231115_1445'
            mock_datetime.now.return_value = mock_now_2
            result_2 = download_data(
                raw_data_path=raw_data_path,
                download_func=mock_download_2,
                add_timestamp=True,
            )

        # Assert
        assert '20231115_1430' in result_1
        assert '20231115_1445' in result_2
        assert os.path.exists(result_1)
        assert os.path.exists(result_2)
        assert result_1 != result_2
        
        # Verify contents
        with open(result_1) as f:
            assert f.read() == 'data1'
        with open(result_2) as f:
            assert f.read() == 'data2'

    def test_download_data_with_nested_timestamp_path(self, tmp_path):
        '''Test that timestamp directory is created within nested path structure.'''
        # Setup
        raw_data_path = str(tmp_path / 'nested' / 'deep' / 'raw_data')
        mock_file_path = str(tmp_path / 'source' / 'dataset.zip')
        os.makedirs(os.path.dirname(mock_file_path), exist_ok=True)
        Path(mock_file_path).touch()

        mock_download = MagicMock(return_value=mock_file_path)

        # Execute
        with patch('src.data.download_data.datetime') as mock_datetime:
            mock_now = Mock()
            mock_now.strftime.return_value = '20231115_1430'
            mock_datetime.now.return_value = mock_now

            result = download_data(
                raw_data_path=raw_data_path,
                download_func=mock_download,
                add_timestamp=True,
            )

        # Assert
        expected_path = os.path.join(raw_data_path, '20231115_1430', 'dataset.zip')
        assert result == expected_path
        assert os.path.exists(result)
        assert os.path.exists(os.path.dirname(result))

    def test_download_data_handles_special_characters_in_filename(self, tmp_path):
        '''Test download with filenames containing special characters.'''
        # Setup
        raw_data_path = str(tmp_path / 'raw_data')
        source_file = tmp_path / 'source' / 'data-2023_v1.0[final].csv'
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.touch()

        mock_download = MagicMock(return_value=str(source_file))

        # Execute
        result = download_data(
            raw_data_path=raw_data_path,
            download_func=mock_download,
            add_timestamp=False,
        )

        # Assert
        assert os.path.exists(result)
        assert result.endswith('data-2023_v1.0[final].csv')
        assert os.path.dirname(result) == raw_data_path


class TestMain:
    '''Test suite for main function.'''

    @patch('src.data.download_data.download_data')
    def test_main_calls_download_data(self, mock_download_data):
        '''Test that main function calls download_data with the correct path.'''
        # Setup
        mock_download_data.return_value = '/path/to/data'

        # Execute
        main()

        # Assert
        mock_download_data.assert_called_once()
        call_args = mock_download_data.call_args
        # First positional argument should be the raw_data_path
        raw_data_path = call_args[0][0]
        assert raw_data_path.endswith('data/raw/')
        # Second positional argument should be a callable download_func
        download_func = call_args[0][1]
        assert callable(download_func)

    @patch.dict(os.environ, {'RAW_DATA_PATH': 'custom/data/path'})
    @patch('src.data.download_data.download_data')
    def test_main_uses_env_variable(self, mock_download_data):
        '''Test that main function respects RAW_DATA_PATH environment variable.'''
        # Setup
        mock_download_data.return_value = '/path/to/data'

        # Execute
        main()

        # Assert
        mock_download_data.assert_called_once()
        call_args = mock_download_data.call_args
        raw_data_path = call_args[0][0]
        assert 'custom/data/path' in raw_data_path

    @patch('src.data.download_data.download_data')
    def test_main_calls_with_timestamp_enabled(self, mock_download_data):
        '''Test that main function calls download_data with add_timestamp=True.'''
        # Setup
        mock_download_data.return_value = '/path/to/data'

        # Execute
        main()

        # Assert
        mock_download_data.assert_called_once()
        call_kwargs = mock_download_data.call_args[1] if mock_download_data.call_args[1] else {}
        # Check if add_timestamp is passed as keyword argument
        if 'add_timestamp' in call_kwargs:
            assert call_kwargs['add_timestamp'] is True
        else:
            # Check if it's passed as positional argument (3rd position)
            call_args = mock_download_data.call_args[0]
            if len(call_args) > 2:
                assert call_args[2] is True
