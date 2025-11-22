"""
OECD Migration Analysis - Revised Data Loader for Available CSV Data
Works with aggregate labor market data and education distribution
"""

import pandas as pd
import numpy as np
import os

class OECDMigrationData:
    """
    Unified data loader for OECD migration CSV files.
    Handles available data: aggregate labor market outcomes, education distribution, migration inflows.
    """
    
    def __init__(self, data_dir='c:/Users/Vedh/Documents/oecd-migration-analysis/data/'):
        self.data_dir = data_dir
        self.files = {
            'educational_attainment': 'adults educational attainment distribution.csv',
            'international_migration': 'international migration database.csv',
            'labor_market': 'labour market outcomes of immigrants.csv',
            'permanent_inflows': 'standardised inflows of permanent-type migrants.csv',
            'temporary_inflows': 'standardised inflows of temporary migrants.csv'
        }
        
        self.raw_data = {}
        self.all_countries = None
        
    def load_all_data(self, verbose=True):
        """Load all CSV files."""
        if verbose:
            print("="*80)
            print("LOADING OECD MIGRATION DATA FROM CSV FILES")
            print("="*80)
        
        for key, filename in self.files.items():
            filepath = os.path.join(self.data_dir, filename)
            if verbose:
                print(f"\nLoading {key}...")
            
            try:
                df = pd.read_csv(filepath)
                self.raw_data[key] = df
                
                if verbose:
                    print(f"  OK Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
                    if 'REF_AREA' in df.columns:
                        n_countries = len(df['REF_AREA'].unique())
                        print(f"  OK Countries: {n_countries}")
            
            except FileNotFoundError:
                if verbose:
                    print(f"  X File not found: {filename}")
                self.raw_data[key] = None
        
        self._extract_all_countries()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"DATA LOADING COMPLETE")
            print(f"Total unique countries: {len(self.all_countries)}")
            print(f"Countries: {', '.join(sorted(self.all_countries)[:15])}...")
            print(f"{'='*80}\n")
        
        return self.raw_data
    
    def _extract_all_countries(self):
        """Extract unique countries from all datasets."""
        countries = set()
        for df in self.raw_data.values():
            if df is not None and 'REF_AREA' in df.columns:
                countries.update(df['REF_AREA'].unique())
        self.all_countries = sorted(list(countries))
    
    def get_labor_force_participation(self, countries=None):
        """
        Get labor force participation rates (aggregate, not by education).
        
        Returns:
        --------
        DataFrame : Labor force participation rates by country, year, birth place
        """
        if 'labor_market' not in self.raw_data:
            raise ValueError("Labor market data not loaded")
        
        df = self.raw_data['labor_market'].copy()
        
        # Filter to labor force participation rate
        df = df[df['MEASURE'] == 'LF_RATE'].copy()
        
        # Create population group
        birth_mapping = {'FB': 'Foreign-born', 'NB': 'Native-born'}
        df['population_group'] = df['BIRTH_PLACE'].map(birth_mapping)
        
        # Filter countries if specified
        if countries is not None:
            df = df[df['REF_AREA'].isin(countries)]
        
        # Keep relevant columns
        result = df[['REF_AREA', 'TIME_PERIOD', 'population_group', 'OBS_VALUE']].copy()
        result = result.rename(columns={
            'REF_AREA': 'country',
            'TIME_PERIOD': 'year',
            'OBS_VALUE': 'lfp_rate'
        })
        
        return result.dropna(subset=['population_group', 'lfp_rate'])
    
    def get_education_distribution(self, countries=None):
        """
        Get education distribution (% of population at each education level).
        
        Returns:
        --------
        DataFrame : Education distribution by country, year, birth place, education level
        """
        if 'educational_attainment' not in self.raw_data:
            raise ValueError("Educational attainment data not loaded")
        
        df = self.raw_data['educational_attainment'].copy()
        
        # Create education categories
        edu_mapping = {
            'ISCED11A_0T2': 'Low',      # Primary/Lower Secondary
            'ISCED11A_3_4': 'Medium',   # Upper Secondary
            'ISCED11A_5T8': 'High'      # Tertiary
        }
        df['edu_cat'] = df['ATTAINMENT_LEV'].map(edu_mapping)
        
        # Create population group
        birth_mapping = {'FB': 'Foreign-born', 'NB': 'Native-born'}
        df['population_group'] = df['BIRTH_PLACE'].map(birth_mapping)
        
        # Filter countries if specified
        if countries is not None:
            df = df[df['REF_AREA'].isin(countries)]
        
        # Keep relevant columns
        result = df[['REF_AREA', 'TIME_PERIOD', 'edu_cat', 'population_group', 'OBS_VALUE']].copy()
        result = result.rename(columns={
            'REF_AREA': 'country',
            'TIME_PERIOD': 'year',
            'OBS_VALUE': 'percentage'
        })
        
        return result.dropna(subset=['edu_cat', 'population_group', 'percentage'])
    
    def get_migration_inflows(self, countries=None):
        """Get total migration inflows."""
        if 'international_migration' not in self.raw_data:
            raise ValueError("International migration data not loaded")
        
        df = self.raw_data['international_migration'].copy()
        
        # Filter to inflows measure
        df = df[df['MEASURE'] == 'B11'].copy()
        
        if countries is not None:
            df = df[df['REF_AREA'].isin(countries)]
        
        inflows = df.groupby(['REF_AREA', 'TIME_PERIOD'])['OBS_VALUE'].sum().reset_index()
        inflows = inflows.rename(columns={
            'REF_AREA': 'country',
            'TIME_PERIOD': 'year',
            'OBS_VALUE': 'total_inflow'
        })
        
        # Add work migration from permanent inflows if available
        if 'permanent_inflows' in self.raw_data and self.raw_data['permanent_inflows'] is not None:
            perm_df = self.raw_data['permanent_inflows'].copy()
            work_df = perm_df[perm_df['MIGRATION_TYPE'] == 'WO'].copy()
            
            if countries is not None:
                work_df = work_df[work_df['REF_AREA'].isin(countries)]
            
            work_summary = work_df.groupby(['REF_AREA', 'TIME_PERIOD'])['OBS_VALUE'].sum().reset_index()
            work_summary = work_summary.rename(columns={
                'REF_AREA': 'country',
                'TIME_PERIOD': 'year',
                'OBS_VALUE': 'work_migration'
            })
            
            inflows = inflows.merge(work_summary, on=['country', 'year'], how='left')
            inflows['work_migration'] = inflows['work_migration'].fillna(0)
            inflows['work_share'] = (inflows['work_migration'] / inflows['total_inflow']) * 100
        
        return inflows
    
    def get_data_summary(self):
        """Get summary of data availability."""
        summary = []
        
        for country in self.all_countries:
            country_info = {'country': country}
            
            for key, df in self.raw_data.items():
                if df is not None and 'REF_AREA' in df.columns:
                    has_data = country in df['REF_AREA'].values
                    country_info[key] = 'Yes' if has_data else 'No'
                else:
                    country_info[key] = 'N/A'
            
            summary.append(country_info)
        
        return pd.DataFrame(summary)


def load_oecd_data(countries=None, verbose=True):
    """
    Quick function to load all OECD migration data.
    
    Returns:
    --------
    dict : Dictionary with processed dataframes
    """
    loader = OECDMigrationData()
    loader.load_all_data(verbose=verbose)
    
    data = {
        'labor_force_participation': loader.get_labor_force_participation(countries),
        'education_distribution': loader.get_education_distribution(countries),
        'migration_inflows': loader.get_migration_inflows(countries),
        'all_countries': loader.all_countries,
        'loader': loader
    }
    
    if verbose:
        print("\nProcessed datasets ready for analysis:")
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                countries_in_data = value['country'].unique() if 'country' in value.columns else []
                print(f"  * {key}: {len(value):,} rows, {len(countries_in_data)} countries")
    
    return data


if __name__ == "__main__":
    print("Testing OECD Data Loader\n")
    data = load_oecd_data(countries=None, verbose=True)
    
    print(f"\n\nSample labor force participation data:")
    print(data['labor_force_participation'].head(10))
    
    print(f"\n\nSample education distribution data:")
    print(data['education_distribution'].head(10))
