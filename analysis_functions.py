"""
OECD Migration Analysis - Revised Analysis Functions
Works with aggregate labor market data and education distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_lfp_gap(lfp_data):
    """
    Calculate labor force participation gap (Native - Foreign).
    
    Parameters:
    -----------
    lfp_data : DataFrame
        Labor force participation data with columns: country, year, population_group, lfp_rate
        
    Returns:
    --------
    DataFrame : LFP gaps by country and year
    """
    # Pivot to get FB and NB side by side
    lfp_pivot = lfp_data.pivot_table(
        index=['country', 'year'],
        columns='population_group',
        values='lfp_rate'
    ).reset_index()
    
    # Calculate gap (Native - Foreign)
    lfp_pivot['lfp_gap'] = lfp_pivot['Native-born'] - lfp_pivot['Foreign-born']
    
    # Calculate ratio (Foreign / Native)
    lfp_pivot['lfp_ratio'] = lfp_pivot['Foreign-born'] / lfp_pivot['Native-born']
    
    return lfp_pivot


def calculate_skill_composition(education_data):
    """
    Calculate skill composition metrics (% high-skilled for FB vs NB).
    
    Returns:
    --------
    DataFrame : Skill metrics by country and year
    """
    # Get high-skilled percentage
    high_skilled = education_data[education_data['edu_cat'] == 'High'].copy()
    
    high_pivot = high_skilled.pivot_table(
        index=['country', 'year'],
        columns='population_group',
        values='percentage'
    ).reset_index()
    
    # Add skill advantage (FB high-skilled - NB high-skilled)
    if 'Foreign-born' in high_pivot.columns and 'Native-born' in high_pivot.columns:
        high_pivot['skill_advantage'] = high_pivot['Foreign-born'] - high_pivot['Native-born']
    
    return high_pivot


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_migration_trends(inflows_df, countries=None, figsize=(15, 5)):
    """Plot migration trends."""
    df = inflows_df.copy()
    
    if countries is None:
        top_countries = df.groupby('country')['total_inflow'].mean().nlargest(10).index.tolist()
        df = df[df['country'].isin(top_countries)]
        title_suffix = " (Top 10 Countries)"
    else:
        df = df[df['country'].isin(countries)]
        title_suffix = ""
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Total inflows
    for country in sorted(df['country'].unique()):
        country_data = df[df['country'] == country].sort_values('year')
        axes[0].plot(country_data['year'], country_data['total_inflow'], 
                    marker='o', label=country, linewidth=2, alpha=0.7)
    
    axes[0].set_title(f'Total Migration Inflows{title_suffix}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Total Inflows (persons)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Plot 2: Work migration share
    if 'work_share' in df.columns:
        df_work = df[df['work_share'].notna()]
        for country in sorted(df_work['country'].unique()):
            country_data = df_work[df_work['country'] == country].sort_values('year')
            if len(country_data) > 0:
                axes[1].plot(country_data['year'], country_data['work_share'], 
                            marker='s', label=country, linewidth=2, alpha=0.7)
        
        axes[1].set_title(f'Work-Related Migration Share{title_suffix}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Year')
        axes[1].set_xlim(2014, 2024)
        axes[1].set_ylabel('Work Migration (%)')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_lfp_comparison(lfp_data, countries, figsize=(14, 6)):
    """
    Plot labor force participation rates for FB vs NB.
    
    Parameters:
    -----------
    lfp_data : DataFrame
        LFP data
    countries : list
        Countries to plot
    """
    df = lfp_data[lfp_data['country'].isin(countries)].copy()
    
    if len(df) == 0:
        print(f"No data for countries {countries}")
        return None
    
    n_countries = len(countries)
    fig, axes = plt.subplots(n_countries, 1, figsize=(figsize[0], figsize[1]*n_countries))
    
    if n_countries == 1:
        axes = [axes]
    
    for idx, country in enumerate(countries):
        country_data = df[df['country'] == country]
        
        for pop_group in ['Foreign-born', 'Native-born']:
            group_data = country_data[country_data['population_group'] == pop_group].sort_values('year')
            if len(group_data) > 0:
                axes[idx].plot(group_data['year'], group_data['lfp_rate'], 
                              marker='o', label=pop_group, linewidth=2.5, markersize=6)
        
        axes[idx].set_title(f'{country}: Labor Force Participation Rates', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Year')
        axes[idx].set_ylabel('LFP Rate (%)')
        axes[idx].legend(loc='best')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(50, 90)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_lfp_gap_trends(lfp_gaps, countries, figsize=(14, 6)):
    """Plot LFP gap trends over time."""
    df = lfp_gaps[lfp_gaps['country'].isin(countries)].copy()
    
    if len(df) == 0:
        print(f"No data for countries {countries}")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for country in sorted(countries):
        country_data = df[df['country'] == country].sort_values('year')
        if len(country_data) > 0:
            ax.plot(country_data['year'], country_data['lfp_gap'], 
                   marker='o', label=country, linewidth=2.5, markersize=6)
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Parity')
    ax.set_title('Labor Force Participation Gap (Native - Foreign)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Gap (percentage points)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_skill_composition(education_data, countries, year=2022, figsize=(12, 6)):
    """
    Plot education distribution for FB vs NB.
    
    Parameters:
    -----------
    education_data : DataFrame
        Education distribution data
    countries : list
        Countries to plot
    year : int
        Year to visualize
    """
    df = education_data[
        (education_data['year'] == year) &
        (education_data['country'].isin(countries))
    ].copy()
    
    if len(df) == 0:
        print(f"No data for year {year} and countries {countries}")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for grouped bar chart
    edu_levels = ['Low', 'Medium', 'High']
    n_countries = len(countries)
    n_edu = len(edu_levels)
    
    bar_width = 0.12
    x_positions = np.arange(n_edu) * 1.0
    
    colors_fb = plt.cm.Paired(np.linspace(0, 0.5, n_countries))
    colors_nb = plt.cm.Paired(np.linspace(0.5, 1, n_countries))
    
    for idx, country in enumerate(sorted(countries)):
        country_data = df[df['country'] == country]
        
        fb_vals = []
        nb_vals = []
        
        for edu in edu_levels:
            fb = country_data[
                (country_data['edu_cat'] == edu) & 
                (country_data['population_group'] == 'Foreign-born')
            ]['percentage'].values
            nb = country_data[
                (country_data['edu_cat'] == edu) & 
                (country_data['population_group'] == 'Native-born')
            ]['percentage'].values
            
            fb_vals.append(fb[0] if len(fb) > 0 else np.nan)
            nb_vals.append(nb[0] if len(nb) > 0 else np.nan)
        
        offset = idx * bar_width * 2
        
        ax.bar(x_positions + offset, fb_vals, bar_width, 
               label=f'{country} - FB', alpha=0.85, color=colors_fb[idx])
        ax.bar(x_positions + bar_width + offset, nb_vals, bar_width, 
               label=f'{country} - NB', alpha=0.6, color=colors_nb[idx],
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Education Level', fontweight='bold', fontsize=12)
    ax.set_ylabel('% of Population', fontweight='bold', fontsize=12)
    ax.set_title(f'Education Distribution: Foreign-Born vs Native-Born ({year})', 
                fontsize=14, fontweight='bold')
    
    center_offset = (n_countries - 1) * bar_width
    ax.set_xticks(x_positions + center_offset)
    ax.set_xticklabels(edu_levels, fontsize=11)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, ncol=2 if n_countries > 4 else 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_highskill_comparison(skill_metrics, countries, figsize=(12, 6)):
    """
    Plot high-skilled percentage comparison.
    
    Parameters:
    -----------
    skill_metrics : DataFrame
        Result from calculate_skill_composition()
    countries : list
        Countries to plot
    """
    df = skill_metrics[skill_metrics['country'].isin(countries)].copy()
    
    if len(df) == 0:
        print(f"No data for countries {countries}")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for country in sorted(countries):
        country_data = df[df['country'] == country].sort_values('year')
        if len(country_data) > 0 and 'Foreign-born' in country_data.columns:
            ax.plot(country_data['year'], country_data['Foreign-born'], 
                   marker='o', label=f'{country} - FB', linewidth=2.5, markersize=6)
            if 'Native-born' in country_data.columns:
                ax.plot(country_data['year'], country_data['Native-born'], 
                       marker='s', label=f'{country} - NB', linewidth=2, markersize=5, 
                       linestyle='--', alpha=0.7)
    
    ax.set_title('High-Skilled Population (%): Foreign-Born vs Native-Born', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('% with Tertiary Education', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2 if len(countries) > 3 else 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_integration_dashboard(lfp_data, education_data, inflows_data,
                               countries, year=2022, figsize=(16, 10)):
    """
    Create comprehensive integration dashboard.
    
    Parameters:
    -----------
    lfp_data : DataFrame
        Labor force participation data
    education_data : DataFrame
        Education distribution data
    inflows_data : DataFrame
        Migration inflow data
    countries : list
        Countries to analyze
    year : int
        Year for snapshot
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. LFP Rates comparison
    ax1 = fig.add_subplot(gs[0, 0])
    lfp_year = lfp_data[
        (lfp_data['year'] == year) &
        (lfp_data['country'].isin(countries))
    ].copy()
    
    if len(lfp_year) > 0:
        lfp_pivot = lfp_year.pivot_table(
            index='country',
            columns='population_group',
            values='lfp_rate'
        )
        lfp_pivot.plot(kind='bar', ax=ax1, width=0.7, alpha=0.8)
        ax1.set_title(f'Labor Force Participation Rates ({year})', fontweight='bold')
        ax1.set_xlabel('Country')
        ax1.set_ylabel('LFP Rate (%)')
        ax1.legend(title='')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    
    # 2. Education distribution
    ax2 = fig.add_subplot(gs[0, 1])
    edu_year = education_data[
        (education_data['year'] == year) &
        (education_data['country'].isin(countries))
    ].copy()
    
    if len(edu_year) > 0:
        high_skilled = edu_year[edu_year['edu_cat'] == 'High']
        if len(high_skilled) > 0:
            hs_pivot = high_skilled.pivot_table(
                index='country',
                columns='population_group',
                values='percentage'
            )
            hs_pivot.plot(kind='bar', ax=ax2, width=0.7, alpha=0.8, color=['#ff7f0e', '#1f77b4'])
            ax2.set_title(f'High-Skilled Population % ({year})', fontweight='bold')
            ax2.set_xlabel('Country')
            ax2.set_ylabel('% with Tertiary Education')
            ax2.legend(title='')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    
    # 3. Migration inflows trend
    ax3 = fig.add_subplot(gs[1, :])
    inflow_recent = inflows_data[inflows_data['country'].isin(countries)].copy()
    inflow_recent = inflow_recent[inflow_recent['year'] >= year - 10]
    
    for country in sorted(countries):
        country_data = inflow_recent[inflow_recent['country'] == country].sort_values('year')
        if len(country_data) > 0:
            ax3.plot(country_data['year'], country_data['total_inflow'], 
                    marker='o', label=country, linewidth=2.5, markersize=6)
    
    ax3.set_title('Migration Inflows (Last 10 Years)', fontweight='bold', fontsize=13)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Total Inflows (persons)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.suptitle(f'Migration Integration Dashboard - {", ".join(countries)}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.show()
    return fig


def create_summary_table(lfp_gaps, skill_metrics, countries, year=2022):
    """Create summary statistics table."""
    print(f"\n{'='*80}")
    print(f"MIGRATION INTEGRATION SUMMARY - {year}")
    print(f"Countries: {', '.join(countries)}")
    print(f"{'='*80}\n")
    
    # LFP summary
    lfp_year = lfp_gaps[
        (lfp_gaps['year'] == year) &
        (lfp_gaps['country'].isin(countries))
    ].copy()
    
    if len(lfp_year) > 0:
        print("Labor Force Participation:")
        print("-" * 80)
        summary_lfp = lfp_year[['country', 'Foreign-born', 'Native-born', 'lfp_gap', 'lfp_ratio']].copy()
        summary_lfp = summary_lfp.set_index('country')
        print(summary_lfp.round(2))
        print(f"\n  * Gap = Native - Foreign (negative means migrants have higher LFP)")
        print(f"  * Ratio = Foreign / Native (1.0 = parity)")
    
    # Skill summary
    skill_year = skill_metrics[
        (skill_metrics['year'] == year) &
        (skill_metrics['country'].isin(countries))
    ].copy()
    
    if len(skill_year) > 0:
        print(f"\n\nHigh-Skilled Population (% with Tertiary Education):")
        print("-" * 80)
        cols = ['country', 'Foreign-born', 'Native-born']
        if 'skill_advantage' in skill_year.columns:
            cols.append('skill_advantage')
        summary_skill = skill_year[cols].copy()
        summary_skill = summary_skill.set_index('country')
        print(summary_skill.round(2))
        if 'skill_advantage' in skill_year.columns:
            print(f"\n  * Skill advantage = FB % - NB % (positive means migrants are more educated)")
    
    print(f"\n{'='*80}\n")
    
    return {'lfp': lfp_year if len(lfp_year) > 0 else None, 
            'skill': skill_year if len(skill_year) > 0 else None}
