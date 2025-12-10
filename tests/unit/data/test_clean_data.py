"""
Unit tests for data cleaning module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.clean_data import (
    merge_raw_tables,
    filter_metropolitan_areas,
    select_and_rename_columns,
    clean_missing_data,
    convert_coordinates,
    add_holiday_column,
    rearrange_severity_values,
    transform_data_pipeline
)


@pytest.fixture
def sample_caracteristics():
    """Sample caracteristics data for testing."""
    return pd.DataFrame({
        'num_acc': [1, 2, 3, 4],
        'an': [2020, 2020, 2021, 2021],
        'mois': [1, 2, 3, 4],
        'jour': [15, 20, 10, 25],
        'hrmn': [1430, 1545, 900, 2130],
        'lum': [1, 2, 1, 3],
        'agg': [1, 2, 1, 2],
        'int': [1, 1, 2, 1],
        'atm': [1, 2, 1, 1],
        'col': [2, 3, 2, 1],
        'com': [75001, 75002, 69001, 13001],
        'adr': ['Rue A', 'Rue B', 'Rue C', 'Rue D'],
        'gps': ['M', 'M', 'A', 'M'],  # Only M should be kept
        'lat': [4883000, 4880000, 4576000, 4330000],
        'long': ['233000', '230000', '483000', '538000'],
        'dep': [75, 75, 69, 13]
    })


@pytest.fixture
def sample_places():
    """Sample places data for testing."""
    return pd.DataFrame({
        'num_acc': [1, 2, 3, 4],
        'catr': [1, 2, 3, 1],
        'voie': [1, 2, 1, 3],
        'v1': [0, 1, 2, 0],
        'v2': ['A', 'B', 'C', 'D'],
        'circ': [1, 2, 1, 2],
        'nbv': [2, 3, 2, 4],
        'pr': [10, 20, 30, 40],
        'pr1': [0, 0, 0, 0],
        'vosp': [0, 1, 0, 0],
        'prof': [1, 2, 1, 3],
        'plan': [1, 1, 2, 1],
        'lartpc': [35, 40, 45, 50],
        'larrout': [70, 80, 70, 100],
        'surf': [1, 2, 1, 1],
        'infra': [0, 0, 1, 0],
        'situ': [1, 1, 1, 2],
        'env1': [1, 1, 2, 1]
    })


@pytest.fixture
def sample_users():
    """Sample users data for testing."""
    return pd.DataFrame({
        'user_id': [1, 2, 3, 4],
        'num_acc': [1, 2, 3, 4],
        'place': [1, 2, 1, 1],
        'catu': [1, 2, 1, 3],
        'grav': [1, 2, 3, 4],
        'sexe': [1, 2, 1, 2],
        'trajet': [1, 2, 3, 1],
        'secu': [1, 2, 1, 1],
        'locp': [0, 1, 0, 0],
        'actp': [0, 0, 1, 0],
        'etatp': [1, 1, 1, 1],
        'an_nais': [1985, 1990, 1975, 2000],
        'num_veh': ['A', 'B', 'C', 'D']
    })


@pytest.fixture
def sample_vehicles():
    """Sample vehicles data for testing."""
    return pd.DataFrame({
        'num_acc': [1, 2, 3, 4],
        'senc': [1, 2, 1, 2],
        'catv': [7, 2, 7, 1],
        'occutc': [1, 2, 1, 1],
        'obs': [0, 0, 1, 0],
        'obsm': [0, 0, 0, 0],
        'choc': [1, 2, 3, 1],
        'manv': [1, 2, 1, 3],
        'num_veh': ['A', 'B', 'C', 'D']
    })


@pytest.fixture
def sample_holidays():
    """Sample holidays data for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'ds': ['2020-01-01', '2020-12-25', '2021-01-01'],
        'holiday': ['New Year', 'Christmas', 'New Year']
    })


@pytest.fixture
def sample_dataframes(sample_caracteristics, sample_places, sample_users, 
                      sample_vehicles, sample_holidays):
    """Combined fixture with all sample dataframes."""
    return {
        'caracteristics': sample_caracteristics,
        'places': sample_places,
        'users': sample_users,
        'vehicles': sample_vehicles,
        'holidays': sample_holidays
    }


class TestMergeRawTables:
    """Tests for merge_raw_tables function."""
    
    def test_merge_returns_dataframe(self, sample_dataframes):
        """Test that merge returns a DataFrame."""
        result = merge_raw_tables(sample_dataframes)
        assert isinstance(result, pd.DataFrame)
    
    def test_merge_preserves_num_acc(self, sample_dataframes):
        """Test that num_acc is preserved in merge."""
        result = merge_raw_tables(sample_dataframes)
        assert 'num_acc' in result.columns
        assert len(result) > 0
    
    def test_merge_combines_columns(self, sample_dataframes):
        """Test that merge combines columns from all tables."""
        result = merge_raw_tables(sample_dataframes)
        
        # Check for columns from different tables
        assert 'an' in result.columns  # from caracteristics
        assert 'catr' in result.columns  # from places
        assert 'catu' in result.columns  # from users


class TestFilterMetropolitanAreas:
    """Tests for filter_metropolitan_areas function."""
    
    def test_filter_keeps_only_m_gps(self, sample_caracteristics):
        """Test that only rows with gps='M' are kept."""
        result = filter_metropolitan_areas(sample_caracteristics)
        
        assert all(result['gps'] == 'M')
        assert len(result) == 3  # Only 3 out of 4 have gps='M'
    
    def test_filter_reduces_row_count(self, sample_caracteristics):
        """Test that filtering reduces row count."""
        initial_count = len(sample_caracteristics)
        result = filter_metropolitan_areas(sample_caracteristics)
        
        assert len(result) < initial_count
    
    def test_filter_handles_empty_result(self):
        """Test that filter handles case with no metropolitan areas."""
        df = pd.DataFrame({
            'num_acc': [1, 2],
            'gps': ['A', 'B']
        })
        
        result = filter_metropolitan_areas(df)
        assert len(result) == 0


class TestSelectAndRenameColumns:
    """Tests for select_and_rename_columns function."""
    
    def test_select_returns_correct_columns(self, sample_dataframes):
        """Test that function returns expected columns."""
        merged = merge_raw_tables(sample_dataframes)
        result = select_and_rename_columns(merged)
        
        expected_columns = ['num_acc', 'raw_user_id', 'Year', 'Month', 
                          'User_category', 'Severity', 'Sex', 'Year_of_birth',
                          'Trip_purpose', 'Security', 'Luminosity', 'Weather',
                          'Type_of_road', 'Road_surface', 'Latitude', 'Longitude']
        
        # Check that all expected columns are present
        for col in expected_columns:
            assert col in result.columns, f"Column {col} not found"
    
    def test_select_renames_columns_correctly(self, sample_dataframes):
        """Test that columns are renamed correctly."""
        merged = merge_raw_tables(sample_dataframes)
        result = select_and_rename_columns(merged)
        
        # Check that French column names are gone
        assert 'an' not in result.columns
        assert 'mois' not in result.columns
        
        # Check that English names exist
        assert 'Year' in result.columns
        assert 'Month' in result.columns


class TestCleanMissingData:
    """Tests for clean_missing_data function."""
    
    def test_clean_removes_missing_values(self):
        """Test that rows with missing values are removed."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [1, 2, 3, 4],
            'col3': [1, np.nan, 3, 4]
        })
        
        result = clean_missing_data(df)
        
        assert len(result) == 2  # Only rows 0 and 3 have no missing values
        assert not result.isnull().any().any()
    
    def test_clean_preserves_complete_rows(self):
        """Test that complete rows are preserved."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        result = clean_missing_data(df)
        
        assert len(result) == len(df)
        assert result.equals(df)


class TestConvertCoordinates:
    """Tests for convert_coordinates function."""
    
    def test_convert_divides_by_100000(self):
        """Test that coordinates are divided by 100000."""
        df = pd.DataFrame({
            'Latitude': [4883000, 4880000],
            'Longitude': ['233000', '230000']
        })
        
        result = convert_coordinates(df)
        
        assert result['Latitude'].iloc[0] == pytest.approx(48.83, rel=0.01)
        assert result['Longitude'].iloc[0] == pytest.approx(2.33, rel=0.01)
    
    def test_convert_handles_dash_values(self):
        """Test that dash values are converted to NaN."""
        df = pd.DataFrame({
            'Latitude': [4883000, '-'],
            'Longitude': ['233000', '-']
        })
        
        result = convert_coordinates(df)
        
        assert not pd.isna(result['Latitude'].iloc[0])
        assert pd.isna(result['Latitude'].iloc[1])
    
    def test_convert_handles_string_numbers(self):
        """Test that string numbers are converted properly."""
        df = pd.DataFrame({
            'Latitude': ['4883000', '4880000'],
            'Longitude': ['233000', '230000']
        })
        
        result = convert_coordinates(df)
        
        assert isinstance(result['Latitude'].iloc[0], (int, float))
        assert isinstance(result['Longitude'].iloc[0], (int, float))


class TestAddHolidayColumn:
    """Tests for add_holiday_column function."""
    
    def test_add_holiday_creates_column(self, sample_holidays):
        """Test that holiday column is created."""
        df = pd.DataFrame({'num_acc': [1, 2, 3]})
        
        result = add_holiday_column(df, sample_holidays)
        
        assert 'holiday' in result.columns
    
    def test_add_holiday_defaults_to_false(self, sample_holidays):
        """Test that holiday defaults to False."""
        df = pd.DataFrame({'num_acc': [1, 2, 3]})
        
        result = add_holiday_column(df, sample_holidays)
        
        # Check that all values are False (default implementation)
        assert all(result['holiday'] == False)


class TestRearrangeSeverityValues:
    """Tests for rearrange_severity_values function."""
    
    def test_rearrange_swaps_2_and_4(self):
        """Test that severity values 2 and 4 are swapped."""
        df = pd.DataFrame({
            'Severity': [1, 2, 3, 4]
        })
        
        result = rearrange_severity_values(df)
        
        # After swap: [1, 4, 3, 2]
        assert result['Severity'].iloc[0] == 1
        assert result['Severity'].iloc[1] == 4
        assert result['Severity'].iloc[2] == 3
        assert result['Severity'].iloc[3] == 2
    
    def test_rearrange_preserves_other_values(self):
        """Test that values 1 and 3 are unchanged."""
        df = pd.DataFrame({
            'Severity': [1, 3, 1, 3]
        })
        
        result = rearrange_severity_values(df)
        
        assert result['Severity'].iloc[0] == 1
        assert result['Severity'].iloc[1] == 3


class TestCleanDataPipeline:
    """Integration tests for the complete transformation pipeline."""
    
    def test_pipeline_returns_dataframe(self, sample_dataframes):
        """Test that pipeline returns a DataFrame."""
        result = transform_data_pipeline(sample_dataframes)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_pipeline_filters_and_cleans(self, sample_dataframes):
        """Test that pipeline applies filtering and cleaning."""
        result = transform_data_pipeline(sample_dataframes)
        
        # Should have fewer rows than original (filtered for M and cleaned)
        original_count = len(sample_dataframes['caracteristics'])
        assert len(result) <= original_count
        
        # Should not have missing values
        assert not result.isnull().any().any()
    
    def test_pipeline_has_expected_columns(self, sample_dataframes):
        """Test that pipeline output has expected columns."""
        result = transform_data_pipeline(sample_dataframes)
        
        expected_columns = ['Year', 'Month', 'Severity', 'Latitude', 'Longitude']
        
        for col in expected_columns:
            assert col in result.columns
    
    def test_pipeline_converts_coordinates(self, sample_dataframes):
        """Test that pipeline converts coordinates to decimal degrees."""
        result = transform_data_pipeline(sample_dataframes)
        
        if len(result) > 0:
            # Coordinates should be in reasonable ranges for France
            # Latitude: ~42-51, Longitude: ~-5 to 10
            assert result['Latitude'].min() > 0
            assert result['Latitude'].max() < 100
            assert result['Longitude'].min() > -100
            assert result['Longitude'].max() < 100


@pytest.mark.parametrize("clear_existing", [True, False])
def test_transform_data_integration(clear_existing, monkeypatch):
    """Integration test for transform_data function."""
    # Mock database connection and operations
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Mock load_raw_data_from_db to return sample data
    sample_data = {
        'caracteristics': pd.DataFrame({
            'num_acc': [1, 2],
            'an': [2020, 2020],
            'mois': [1, 2],
            'hrmn': [1430, 715],
            'lum': [1, 2],
            'atm': [1, 1],
            'gps': ['M', 'M'],
            'lat': [4883000, 4880000],
            'long': ['233000', '230000']
        }),
        'places': pd.DataFrame({
            'num_acc': [1, 2],
            'catr': [1, 2],
            'surf': [1, 1]
        }),
        'users': pd.DataFrame({
            'user_id': [1, 2],
            'num_acc': [1, 2],
            'catu': [1, 2],
            'grav': [1, 2],
            'sexe': [1, 2],
            'an_nais': [1985, 1990],
            'trajet': [1, 2],
            'secu': [1, 2]
        }),
        'vehicles': pd.DataFrame({'num_acc': [1, 2]}),
        'holidays': pd.DataFrame({'id': [1], 'ds': ['2020-01-01'], 'holiday': ['New Year']})
    }
    
    # Mock insert_preprocessed_data to return predictable counts
    mock_insert_result = {
        'inserted': 2,
        'updated': 0,
        'unchanged': 0
    }
    
    # Use patch.multiple to reduce nesting
    with patch.multiple('src.data.clean_data',
                        get_db_connection=MagicMock(return_value=mock_conn),
                        load_raw_data_from_db=MagicMock(return_value=sample_data),
                        insert_preprocessed_data=MagicMock(return_value=mock_insert_result),
                        clear_preprocessed_data=MagicMock()):
        from src.data.clean_data import transform_data
        
        result = transform_data(clear_existing=clear_existing)
        
        # Verify result structure and values
        assert isinstance(result, dict)
        
        # Assert all expected keys are present
        assert 'success' in result
        assert 'rows_processed' in result
        assert 'rows_inserted' in result
        assert 'rows_updated' in result
        assert 'rows_unchanged' in result
        assert 'message' in result
        
        # Assert values match expectations
        assert result['success'] is True
        assert result['rows_processed'] == 2  # 2 rows in sample data
        assert result['rows_inserted'] == 2
        assert result['rows_updated'] == 0
        assert result['rows_unchanged'] == 0
        assert result['message'] == 'Data transformation completed successfully'
