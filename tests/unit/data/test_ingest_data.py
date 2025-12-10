import os
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, call
import pytest
import pandas as pd
import psycopg2

from src.data.ingest_data import (
    get_db_connection,
    find_latest_csv_directory,
    load_csv_to_dataframe,
    store_caracteristics,
    store_holidays,
    store_places,
    store_users,
    store_vehicles,
    ingest_data_full,
    ingest_data_chunk
)


class TestGetDbConnection:
    """Test suite for get_db_connection function."""

    @patch.dict(os.environ, {
        'POSTGRES_HOST': 'testhost',
        'POSTGRES_HOST_PORT': '5433',
        'POSTGRES_DB': 'testdb',
        'POSTGRES_USER': 'testuser',
        'POSTGRES_PASSWORD': 'testpass'
    })
    @patch('psycopg2.connect')
    def test_get_db_connection_success(self, mock_connect):
        """Test successful database connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        result = get_db_connection()

        assert result == mock_conn
        mock_connect.assert_called_once_with(
            host='testhost',
            port='5433',
            database='testdb',
            user='testuser',
            password='testpass'
        )

    @patch.dict(os.environ, {
        'POSTGRES_USER': '',
        'POSTGRES_PASSWORD': 'testpass'
    })
    def test_get_db_connection_missing_credentials(self):
        """Test that missing credentials raises ValueError."""
        with pytest.raises(ValueError, match="POSTGRES_USER and POSTGRES_PASSWORD must be set"):
            get_db_connection()

    @patch.dict(os.environ, {
        'POSTGRES_USER': 'testuser',
        'POSTGRES_PASSWORD': 'testpass'
    })
    @patch('psycopg2.connect')
    def test_get_db_connection_failure(self, mock_connect):
        """Test database connection failure."""
        mock_connect.side_effect = psycopg2.Error("Connection failed")

        with pytest.raises(psycopg2.Error):
            get_db_connection()


class TestFindLatestCsvDirectory:
    """Test suite for find_latest_csv_directory function."""

    def test_find_latest_csv_directory_success(self, tmp_path):
        """Test finding the latest CSV directory."""
        # Create directory structure with CSV files
        date_dir1 = tmp_path / '20251120_1000'
        date_dir2 = tmp_path / '20251121_1318'
        subdir1 = date_dir1 / '1'
        subdir2 = date_dir2 / '2'
        
        subdir1.mkdir(parents=True)
        subdir2.mkdir(parents=True)
        
        # Create CSV files
        (subdir1 / 'test.csv').touch()
        (subdir2 / 'test.csv').touch()
        
        # Make subdir2 more recent
        import time
        time.sleep(0.01)
        (subdir2 / 'test.csv').touch()

        result = find_latest_csv_directory(str(tmp_path))

        assert result == subdir2

    def test_find_latest_csv_directory_no_csv_files(self, tmp_path):
        """Test when no CSV files are found."""
        date_dir = tmp_path / '20251121_1318'
        subdir = date_dir / '2'
        subdir.mkdir(parents=True)

        result = find_latest_csv_directory(str(tmp_path))

        assert result is None

    def test_find_latest_csv_directory_nonexistent_path(self):
        """Test with non-existent path."""
        result = find_latest_csv_directory('/nonexistent/path')

        assert result is None


class TestLoadCsvToDataframe:
    """Test suite for load_csv_to_dataframe function."""

    def test_load_csv_to_dataframe_success(self, tmp_path):
        """Test successfully loading a CSV file."""
        csv_file = tmp_path / 'test.csv'
        csv_file.write_text('col1,col2\n1,2\n3,4\n')

        result = load_csv_to_dataframe(csv_file)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['col1', 'col2']

    def test_load_csv_to_dataframe_latin1_fallback(self, tmp_path):
        """Test fallback to latin-1 encoding."""
        csv_file = tmp_path / 'test.csv'
        # Write with latin-1 encoding
        csv_file.write_bytes('col1,col2\ncafé,résumé\n'.encode('latin-1'))

        result = load_csv_to_dataframe(csv_file, encoding='utf-8')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_load_csv_to_dataframe_file_not_found(self):
        """Test with non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            load_csv_to_dataframe(Path('/nonexistent/file.csv'))


class TestStoreCaracteristics:
    """Test suite for store_caracteristics function."""

    def test_store_caracteristics_success(self):
        """Test successfully storing caracteristics data."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.mogrify = MagicMock(return_value=b"INSERT INTO ...")
        mock_conn.cursor.return_value = mock_cursor

        df = pd.DataFrame({
            'Num_Acc': [201600000001, 201600000002],
            'an': [2016, 2016],
            'mois': [1, 1],
            'jour': [1, 2],
            'hrmn': [1200, 1300],
            'lum': [1, 2],
            'agg': [1, 1],
            'int': [1, 1],
            'atm': [1, 1],
            'col': [1, 2],
            'com': [75001, 75002],
            'adr': ['Rue A', 'Rue B'],
            'gps': ['M', 'M'],
            'lat': [48000000, 48100000],
            'long': ['2000000', '2100000'],
            'dep': [75, 75]
        })

        result = store_caracteristics(mock_conn, df)

        assert result == 2
        assert mock_cursor.execute.called
        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()

    def test_store_caracteristics_error_handling(self):
        """Test error handling during data storage."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.mogrify = MagicMock(return_value=b"INSERT INTO ...")
        mock_cursor.execute.side_effect = Exception("Database error")
        mock_conn.cursor.return_value = mock_cursor

        df = pd.DataFrame({
            'Num_Acc': [201600000001],
            'an': [2016],
            'mois': [1],
            'jour': [1],
            'hrmn': [1200],
            'lum': [1],
            'agg': [1],
            'int': [1],
            'atm': [1],
            'col': [1],
            'com': [75001],
            'adr': ['Rue A'],
            'gps': ['M'],
            'lat': [48000000],
            'long': ['2000000'],
            'dep': [75]
        })

        with pytest.raises(Exception):
            store_caracteristics(mock_conn, df)

        mock_conn.rollback.assert_called_once()
        mock_cursor.close.assert_called_once()


class TestStoreHolidays:
    """Test suite for store_holidays function."""

    def test_store_holidays_success(self):
        """Test successfully storing holidays data."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.mogrify = MagicMock(return_value=b"INSERT INTO ...")
        mock_conn.cursor.return_value = mock_cursor

        df = pd.DataFrame({
            'ds': ['2016-01-01', '2016-12-25'],
            'holiday': ['New Year', 'Christmas']
        })

        result = store_holidays(mock_conn, df)

        assert result == 2
        mock_conn.commit.assert_called_once()


class TestStorePlaces:
    """Test suite for store_places function."""

    def test_store_places_success(self):
        """Test successfully storing places data."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.mogrify = MagicMock(return_value=b"INSERT INTO ...")
        mock_conn.cursor.return_value = mock_cursor

        df = pd.DataFrame({
            'Num_Acc': [201600000001, 201600000002],
            'catr': [1, 2],
            'voie': [1, 1],
            'v1': [1, 1],
            'v2': ['A', 'B'],
            'circ': [1, 2],
            'nbv': [2, 2],
            'pr': [100, 200],
            'pr1': [0, 0],
            'vosp': [1, 1],
            'prof': [1, 1],
            'plan': [1, 1],
            'lartpc': [10, 10],
            'larrout': [7, 7],
            'surf': [1, 1],
            'infra': [0, 0],
            'situ': [1, 1],
            'env1': [1, 1]
        })

        result = store_places(mock_conn, df)

        assert result == 2
        mock_conn.commit.assert_called_once()


class TestStoreUsers:
    """Test suite for store_users function."""

    def test_store_users_success(self):
        """Test successfully storing users data."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.mogrify = MagicMock(return_value=b"INSERT INTO ...")
        mock_conn.cursor.return_value = mock_cursor

        df = pd.DataFrame({
            'Num_Acc': [201600000001, 201600000001],
            'place': [1, 1],
            'catu': [1, 1],
            'grav': [1, 3],
            'sexe': [2, 1],
            'trajet': [0, 9],
            'secu': [11, 21],
            'locp': [0, 0],
            'actp': [0, 0],
            'etatp': [0, 0],
            'an_nais': [1983, 2001],
            'num_veh': ['B02', 'A01']
        })

        result = store_users(mock_conn, df)

        assert result == 2
        mock_conn.commit.assert_called_once()


class TestStoreVehicles:
    """Test suite for store_vehicles function."""

    def test_store_vehicles_success(self):
        """Test successfully storing vehicles data."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.mogrify = MagicMock(return_value=b"INSERT INTO ...")
        mock_conn.cursor.return_value = mock_cursor

        df = pd.DataFrame({
            'Num_Acc': [201600000001, 201600000001],
            'senc': [1, 1],
            'catv': [7, 7],
            'occutc': [1, 1],
            'obs': [0, 0],
            'obsm': [0, 0],
            'choc': [1, 2],
            'manv': [0, 0],
            'num_veh': ['A01', 'B02']
        })

        result = store_vehicles(mock_conn, df)

        assert result == 2
        mock_conn.commit.assert_called_once()


class TestIngestDataFull:
    """Test suite for ingest_data_full function."""

    def test_ingest_data_full_success(self, tmp_path):
        """Test the complete ingest_data_full workflow."""
        # Create test CSV directory structure
        date_dir = tmp_path / '20251121_1318'
        csv_dir = date_dir / '2'
        csv_dir.mkdir(parents=True)

        # Create minimal test CSV files
        (csv_dir / 'caracteristics.csv').write_text(
            'Num_Acc,an,mois,jour,hrmn,lum,agg,int,atm,col,com,adr,gps,lat,long,dep\n'
            '201600000001,2016,1,1,1200,1,1,1,1,1,75001,Rue A,M,48000000,2000000,75\n'
        )
        (csv_dir / 'holidays.csv').write_text(
            'ds,holiday\n2016-01-01,New Year\n'
        )
        (csv_dir / 'places.csv').write_text(
            'Num_Acc,catr,voie,v1,v2,circ,nbv,pr,pr1,vosp,prof,plan,lartpc,larrout,surf,infra,situ,env1\n'
            '201600000001,1,1,1,A,1,2,100,0,1,1,1,10,7,1,0,1,1\n'
        )
        (csv_dir / 'users.csv').write_text(
            'Num_Acc,place,catu,grav,sexe,trajet,secu,locp,actp,etatp,an_nais,num_veh\n'
            '201600000001,1,1,1,2,0,11,0,0,0,1983,B02\n'
        )
        (csv_dir / 'vehicles.csv').write_text(
            'Num_Acc,senc,catv,occutc,obs,obsm,choc,manv,num_veh\n'
            '201600000001,1,7,1,0,0,1,0,A01\n'
        )

        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.mogrify = MagicMock(return_value=b"INSERT INTO ...")
        mock_conn.cursor.return_value = mock_cursor

        result = ingest_data_full(
            raw_data_path=str(tmp_path),
            db_connection=mock_conn
        )

        assert 'caracteristics' in result
        assert 'holidays' in result
        assert 'places' in result
        assert 'users' in result
        assert 'vehicles' in result
        assert result['caracteristics'] >= 0
        mock_conn.commit.call_count >= 5  # At least one commit per table

    def test_ingest_data_full_missing_csv_files(self, tmp_path):
        """Test when CSV files are missing."""
        date_dir = tmp_path / '20251121_1318'
        csv_dir = date_dir / '2'
        csv_dir.mkdir(parents=True)

        # Create only one CSV file
        (csv_dir / 'caracteristics.csv').write_text('Num_Acc\n1\n')

        mock_conn = MagicMock()

        with pytest.raises(ValueError, match="Missing CSV files"):
            ingest_data_full(raw_data_path=str(tmp_path), db_connection=mock_conn)

    def test_ingest_data_full_no_csv_directory(self, tmp_path):
        """Test when no CSV directory is found."""
        mock_conn = MagicMock()

        with pytest.raises(ValueError, match="No CSV files found"):
            ingest_data_full(raw_data_path=str(tmp_path), db_connection=mock_conn)

    def test_ingest_data_full_empty_raw_data_path(self):
        """Test with empty raw_data_path."""
        with patch.dict(os.environ, {'DATA_RAW_PATH': ''}):
            with pytest.raises(ValueError, match="raw_data_path must be provided"):
                ingest_data_full(raw_data_path='')

class TestLoadNextChunk:
    """Test suite for ingest_data_chunk function."""

    def test_load_next_chunk_success(self, tmp_path):
        """Test the complete ingest_data_chunk workflow."""
        # Create test CSV directory structure
        date_dir = tmp_path / '20251121_1318'
        csv_dir = date_dir / '2'
        csv_dir.mkdir(parents=True)

        # Create minimal test CSV files
        (csv_dir / 'caracteristics.csv').write_text(
            'Num_Acc,an,mois,jour,hrmn,lum,agg,int,atm,col,com,adr,gps,lat,long,dep\n'
            '201600000001,2016,1,1,1200,1,1,1,1,1,75001,Rue A,M,48000000,2000000,75\n'
        )
        (csv_dir / 'holidays.csv').write_text(
            'ds,holiday\n2016-01-01,New Year\n'
        )
        (csv_dir / 'places.csv').write_text(
            'Num_Acc,catr,voie,v1,v2,circ,nbv,pr,pr1,vosp,prof,plan,lartpc,larrout,surf,infra,situ,env1\n'
            '201600000001,1,1,1,A,1,2,100,0,1,1,1,10,7,1,0,1,1\n'
        )
        (csv_dir / 'users.csv').write_text(
            'Num_Acc,place,catu,grav,sexe,trajet,secu,locp,actp,etatp,an_nais,num_veh\n'
            '201600000001,1,1,1,2,0,11,0,0,0,1983,B02\n'
        )
        (csv_dir / 'vehicles.csv').write_text(
            'Num_Acc,senc,catv,occutc,obs,obsm,choc,manv,num_veh\n'
            '201600000001,1,7,1,0,0,1,0,A01\n'
        )

        # Mock database connection and operations
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.mogrify = MagicMock(return_value=b"INSERT INTO ...")
        mock_cursor.fetchall.return_value = []  # No existing progress
        mock_conn.cursor.return_value = mock_cursor

        # Mock progress tracking to simulate initialized state
        csv_dir_str = str(csv_dir)
        mock_progress = {
            'caracteristics': {
                'table_name': 'caracteristics',
                'csv_directory': csv_dir_str,
                'total_rows': 1,
                'rows_loaded': 0,
                'chunk_size': 1000,
                'is_complete': False,
                'progress_percentage': 0.0
            },
            'holidays': {
                'table_name': 'holidays',
                'csv_directory': csv_dir_str,
                'total_rows': 1,
                'rows_loaded': 0,
                'chunk_size': 1000,
                'is_complete': False,
                'progress_percentage': 0.0
            },
            'places': {
                'table_name': 'places',
                'csv_directory': csv_dir_str,
                'total_rows': 1,
                'rows_loaded': 0,
                'chunk_size': 1000,
                'is_complete': False,
                'progress_percentage': 0.0
            },
            'users': {
                'table_name': 'users',
                'csv_directory': csv_dir_str,
                'total_rows': 1,
                'rows_loaded': 0,
                'chunk_size': 1000,
                'is_complete': False,
                'progress_percentage': 0.0
            },
            'vehicles': {
                'table_name': 'vehicles',
                'csv_directory': csv_dir_str,
                'total_rows': 1,
                'rows_loaded': 0,
                'chunk_size': 1000,
                'is_complete': False,
                'progress_percentage': 0.0
            }
        }

        with patch('src.data.ingest_data.get_db_connection', return_value=mock_conn), \
             patch('src.data.ingest_data.get_progress_status', return_value=mock_progress), \
             patch('src.data.ingest_data.update_progress'), \
             patch('src.data.ingest_data.initialize_progress_tracking'):
            result = ingest_data_chunk(
                raw_data_path=str(tmp_path),
                db_connection=mock_conn
            )
        assert 'caracteristics' in result
        assert 'holidays' in result
        assert 'places' in result
        assert 'users' in result
        assert 'vehicles' in result
        assert result['caracteristics'] >= 0
        mock_conn.commit.call_count >= 5  # At least one commit per table

