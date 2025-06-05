# HESS System ID {{system_id}} Analysis Guide

This document provides comprehensive context for analyzing data from HESS System ID {{system_id}}, enabling sophisticated analytical reasoning and Python-based analyses using pandas. It supports insightful, research-grounded responses based on validated operational patterns derived from extensive field data (Figgener et al., 2024).

## 1. System Overview & Experiment Context
System ID: {{system_id}} (Research-Validated Analysis Framework)
Analysis Goal: To evaluate AI-powered analytical reasoning on HESS operational data using research-validated benchmarks and sophisticated hypothesis-driven analysis.
Primary Dataset: Represents data from {{month}} {{year}} (31 days).
Data Resolution: 1-second intervals (research-standard for accurate analysis).
User Location: Germany.
Electricity Tariff Assumption: Simple Time-of-Use: Peak £0.40/kWh (4 PM-7 PM Mon-Fri), Off-Peak £0.15/kWh (other times). (Use only if query specifically asks about cost).

## 2. Data Structure
**Available DataFrame**: `df` with these key columns:
- `Timestamp`: Primary datetime column (1-second resolution)
- `power_watts`: Power flow (+charging, -discharging)
- `voltage_volts`: Battery pack voltage
- `current_amperes`: Current flow (+charging, -discharging)
- `temp_battery_celsius`: Battery temperature
- `temp_room_celsius`: Ambient temperature
- `interpolated`: Data quality flag (0=measured, 1=interpolated)

**Data Quality Standards (Research-Validated):**
- High-resolution (1-second) measurements essential for accurate analysis
- Low interpolation rates (<1%) indicate excellent data quality
- Large time gaps (>5 minutes) can affect energy calculations significantly
- Missing data patterns may indicate system or measurement issues requiring investigation

## 3. Core Battery Specifications (System ID {{system_id}} - Research Context)
Capacity nominal in Ah: {{Capacity_nominal_in_Ah}}
Voltage nominal in V: {{Voltage_nominal_in_V}}
Energy nominal in kWh: {{Energy_nominal_in_kWh}}
Energy usable datasheet in kWh: {{Energy_usable_datasheet_in_kWh}}
Cell number in series: {{Cell_number_in_series}}
Cell number in parallel: {{Cell_number_in_parallel}}
Cell number: {{Cell_number}}
Inverter nominal power: {{Inverter_nominal_power}} 
Manufacturer: {{Manufacturer}}
Chemistry: {{Chemistry}}
Chemistry detail: {{Chemistry_detail}}
Date storage system installation: {{Date_storage_system_installation}} 
Date measurement system installation: {{Date_measurement_system_installation}}
System Age: {{System_age}} months
Capacity test dates: {{Capacity_test_dates}}

## 4. Normal Operating Ranges
### Battery Chemistry: {{Chemistry}}
{% if Chemistry == "LFP" %}
- **Voltage Range**: 2.5-3.8V per cell ({{Cell_number_in_series}} cells = {{Cell_number_in_series * 2.5}}-{{Cell_number_in_series * 3.8}}V pack)
- **Characteristics**: Flat voltage curve, stable degradation ~2.0%/year
{% elif Chemistry == "NMC" %}
- **Voltage Range**: 3.0-4.2V per cell ({{Cell_number_in_series}} cells = {{Cell_number_in_series * 3.0}}-{{Cell_number_in_series * 4.2}}V pack)
- **Characteristics**: Clear voltage transitions, degradation ~1.9%/year
{% elif Chemistry == "LMO" %}
- **Voltage Range**: 3.0-4.2V per cell ({{Cell_number_in_series}} cells = {{Cell_number_in_series * 3.0}}-{{Cell_number_in_series * 4.2}}V pack)
- **Characteristics**: Clear end-of-charge at ~4.15V/cell, faster degradation ~3.1%/year
{% endif %}

### Temperature Ranges
- **Optimal**: 15-35°C
- **Warning**: Above 40°C
- **Critical**: Above 50°C

### Power & Current (System-Specific)
- **Max Power**: ~{{Inverter_nominal_power * 1000}}W (inverter limited)
- **Max Current**: ~{{(Inverter_nominal_power * 1000 / Voltage_nominal_in_V)|round(1)}}A
- **Typical C-Rate**: 0.05-0.2C ({{(Inverter_nominal_power * 1000 / Voltage_nominal_in_V / Capacity_nominal_in_Ah)|round(2)}}C max)
- **Significant Current Threshold**: >{{(Capacity_nominal_in_Ah * 0.02)|round(2)}}A (2% of capacity)
- **Active Power Threshold**: >{{(Inverter_nominal_power * 10)|round(0)}}W (1% of inverter rating)

## 5. Typical Operational Patterns
### Daily Cycle (Expected)
- **Morning (6-10h)**: Discharge for household use
- **Midday (10-16h)**: Charge from solar (weather dependent)
- **Evening (16-22h)**: Primary discharge period
- **Night (22-6h)**: Minimal activity

### Operational States (Research Benchmark)
- **Idle**: 80-85% of time
- **Charging**: 8-10% of time
- **Discharging**: 8-10% of time

## 6. Common Analysis Tasks

### 6.1. Energy Analysis (Primary Method)
```python
# Calculate daily energy throughput
df_analysis = df.copy()
df_analysis['energy_Wh'] = df_analysis['power_watts'] / 3600  # Convert to Wh per second

# Separate charging and discharging
charge_energy = df_analysis[df_analysis['power_watts'] > 0]['energy_Wh'].sum()
discharge_energy = abs(df_analysis[df_analysis['power_watts'] < 0]['energy_Wh'].sum())
efficiency = (discharge_energy / charge_energy * 100) if charge_energy > 0 else 0
```

### 6.2. Operational State Analysis (Research-Validated Thresholds)
```python
# System-specific thresholds based on capacity
capacity_ah = {{Capacity_nominal_in_Ah}}
current_threshold = capacity_ah * 0.02  # 2% of capacity (research-validated)
power_threshold = {{Inverter_nominal_power * 10}}  # 1% of inverter power

# Method 1: Power-based (recommended for energy analysis)
df['state_power'] = 'idle'
df.loc[df['power_watts'] > power_threshold, 'state_power'] = 'charging'
df.loc[df['power_watts'] < -power_threshold, 'state_power'] = 'discharging'

# Method 2: Current-based (recommended for detailed electrical analysis)
df['state_current'] = 'idle'
df.loc[df['current_amperes'] > current_threshold, 'state_current'] = 'charging'
df.loc[df['current_amperes'] < -current_threshold, 'state_current'] = 'discharging'

# Calculate state distribution (should match research: ~80-85% idle)
state_distribution_power = df['state_power'].value_counts(normalize=True) * 100
state_distribution_current = df['state_current'].value_counts(normalize=True) * 100
```

### 6.3. Temperature Analysis
```python
# Check temperature performance
avg_battery_temp = df['temp_battery_celsius'].mean()
max_battery_temp = df['temp_battery_celsius'].max()
time_above_optimal = (df['temp_battery_celsius'] > 35).sum() / len(df) * 100
time_warning = (df['temp_battery_celsius'] > 40).sum() / len(df) * 100
```

### 6.4. Voltage Analysis (Chemistry-Specific)
```python
# Voltage distribution analysis
voltage_stats = {
    'mean': df['voltage_volts'].mean(),
    'min': df['voltage_volts'].min(),
    'max': df['voltage_volts'].max(),
    'std': df['voltage_volts'].std()
}

# Check against expected range for {{Chemistry}}
{% if Chemistry == "LFP" %}
expected_min, expected_max = {{Cell_number_in_series * 2.5}}, {{Cell_number_in_series * 3.8}}
{% elif Chemistry == "NMC" %}
expected_min, expected_max = {{Cell_number_in_series * 3.0}}, {{Cell_number_in_series * 4.2}}
{% elif Chemistry == "LMO" %}
expected_min, expected_max = {{Cell_number_in_series * 3.0}}, {{Cell_number_in_series * 4.2}}
{% endif %}

voltage_range_check = {
    'within_expected': (voltage_stats['min'] >= expected_min * 0.95) and (voltage_stats['max'] <= expected_max * 1.05),
    'expected_range': f"{expected_min:.1f}V - {expected_max:.1f}V",
    'actual_range': f"{voltage_stats['min']:.1f}V - {voltage_stats['max']:.1f}V"
}
```

### 6.5. Data Quality Assessment
```python
# Check data quality indicators
interpolated_percentage = (df['interpolated'] == 1).sum() / len(df) * 100
missing_data = df.isnull().sum()
time_gaps = df['Timestamp'].diff().dt.total_seconds()
large_gaps = (time_gaps > 300).sum()  # Gaps > 5 minutes

data_quality = {
    'interpolated_percent': interpolated_percentage,
    'quality_rating': 'Excellent' if interpolated_percentage < 1 else 'Good' if interpolated_percentage < 5 else 'Poor',
    'large_time_gaps': large_gaps,
    'missing_values': missing_data.sum()
}
```

## 7. Health Assessment Guidelines
### Good Indicators
- Round-trip efficiency: 85-95%
- Temperature mostly 15-35°C
- Voltage within expected range for chemistry
- Operational states match research benchmarks (80-85% idle)
- Low interpolation rate (<1%)

### Warning Signs
- Efficiency < 80%
- Frequent high temperatures (>40°C)
- Voltage consistently outside normal range
- Unusual operational state distribution (e.g., <70% idle time)
- High interpolation rate (>5%)

## 8. Analysis Limitations
- **30-day data**: Good for operational patterns, limited for degradation assessment
- **Capacity estimation**: Requires full charge/discharge cycles for accuracy
- **Seasonal effects**: {{month}} data may not represent year-round performance

## 9. Key Formulas
- **Energy**: Power × Time (integrate power over time for total energy)
- **Efficiency**: Energy_out / Energy_in × 100%
- **C-Rate**: Current / Capacity (e.g., 15A / 15Ah = 1C)
- **State of Charge**: Estimate from voltage (chemistry-dependent)

Remember: Always check data quality (interpolated values, missing data) before analysis.

## 10. Research-Validated Operational Patterns (Figgener et al., 2024)

### 10.1. Normal Operational State Distribution
**Research Benchmark (106 System-Years):**
- **Idle: 80-85%** of operational time (systems not actively charging/discharging)
- **Charging: 8-10%** of operational time  
- **Discharging: 8-10%** of operational time

**Interpretation Guidelines:**
- Significant deviations may indicate: undersized battery, oversized battery, or control strategy issues
- Higher discharge percentage: Possible battery undersizing relative to consumption
- Higher charging percentage: Possible battery undersizing relative to PV generation
- Very high idle percentage: Possible battery oversizing

### 10.2. Annual Cycling Behavior
**Research-Validated Expectations:**
- **Small LMO systems:** ~250 equivalent full cycles (EFCs) per year
- **Medium NMC/LFP systems:** ~200 EFCs per year
- **Most cycles:** <20% depth of discharge (shallow cycling is normal and healthy)
- **Occasional deep cycles:** Near 100% DoD (also normal, provides balancing)

### 10.3. Seasonal Cycling Patterns (Northern Hemisphere)
- **Spring/Fall:** Regular full cycles, optimal charge/discharge balance
- **Summer:** Frequent full charges, potential PV overgeneration, reduced discharge frequency
- **Winter:** Incomplete charges, higher grid dependency, reduced PV input

### 10.4. Daily Operational Patterns (Research-Validated)
**Expected Daily Operation Patterns:**
- **Morning (06:00-10:00):** Discharge for household consumption (~2-5 kWh typical)
- **Midday (10:00-16:00):** Charging from PV generation (weather dependent)
- **Evening (16:00-22:00):** Primary discharge period for peak consumption
- **Night (22:00-06:00):** Minimal activity, standby consumption only

## 11. Chemistry-Specific Behavior Patterns

### 11.1. LFP Systems (Lithium Iron Phosphate)
- **Voltage Curves:** Flat OCV curves → difficult SOC estimation from voltage alone
- **Data Distribution:** Less distinct EOC/EOD voltage peaks in data
- **Degradation:** Stable degradation ~2.0% per year
- **Control Implications:** May require different EOC/EOD detection strategies

### 11.2. NMC Systems (Nickel Manganese Cobalt)
- **Voltage Curves:** Clear voltage transitions → easier SOC estimation
- **Data Distribution:** Distinct EOC/EOD voltage peaks visible
- **Degradation:** Moderate degradation ~1.9% per year
- **Variations:** Different NMC compositions show different voltage patterns

### 11.3. LMO Systems (Lithium Manganese Oxide)
- **Voltage Behavior:** Clear EOC voltage around 4.15V, EOD voltage may decrease over time
- **Cycling:** Higher cycle frequency tolerance
- **Degradation:** Faster degradation ~3.1% per year
- **BMS Adaptation:** EOD voltage may decrease over time as BMS compensation for aging

## 12. C-Rate and Power Behavior Analysis

### 12.1. Charging vs Discharging Patterns
**Research Observations:**
- **Charging C-rates:** Typically HIGHER (due to available PV power exceeding household load)
- **Discharging C-rates:** Typically LOWER (nighttime household consumption patterns)
- **System Design Impact:** Smaller systems (higher inverter-to-battery ratio) show higher C-rates

### 12.2. Normal C-Rate Ranges
- **Observed Maximum:** ~0.2C in typical field conditions
- **Most Operation:** Much lower than maximum system capability
- **Design Limitation:** Maximum C-rate typically limited by inverter power, not battery capability

## 13. Advanced Voltage Behavior Interpretation

### 13.1. Voltage Distribution Analysis
**Interpretation Guidelines:**
- **Wide peaks at top/bottom of distribution:** Indicates frequent full charge/discharge states (normal)
- **Multiple peaks in NMC systems:** Different systems have different BMS voltage settings
- **Flat distribution in LFP:** Due to inherent flat OCV characteristic

### 13.2. Voltage Evolution Over System Lifetime
- **EOD Voltage Decrease:** May decrease over system lifetime (BMS aging compensation)
- **EOC Voltage Stability:** Typically more stable than EOD voltage
- **Software Updates:** Can change voltage thresholds and operational behavior

## 14. Capacity Test Framework

### 14.1. Capacity Testing Methodology & Validation Context

**Capacity Test Method by (Figgener et al., 2024):**
- **Gold Standard Procedure:** Full charge → Full discharge at maximum power
- **Field Implementation:** ~2 days per test (preparation, execution, analysis)
- **Validation Results:** 60 manual field tests over 8 years for method validation
- **Purpose:** Validate operational data-based estimation methods

The following capacity tests dates describe the start of capacity test method carried out on the system to calculate the SOH at that point in time with the procedure mentioned before. Given a Systems dataset if it is one of one of the capacity test dates then carry out SOH estimation.

**Research-Validated Degradation Rates:**
- Small LMO systems: 2.1% capacity decrease per year
- Medium NMC systems: 3.2% capacity decrease per year  
- Medium LFP systems: 2.2% capacity decrease per year

### 14.2. What 30-Day Operational Data CAN Determine
**High Confidence Analysis:**
- Temperature performance vs. optimal ranges
- Voltage operation vs. expected ranges for chemistry
- Energy throughput patterns and operational state distribution
- Daily and weekly operational patterns
- Data quality assessment and interpolation rates
- Basic system responsiveness and control behavior

## 15. Fault Detection and Anomaly Recognition

### 15.1. Research-Based Anomaly Detection
**Critical Indicators:**
- **Voltage Anomalies:** Consistently outside chemistry-appropriate ranges, sudden drops under load
- **Temperature Issues:** Outside optimal [15°C, 35°C] or above warning (40°C)
- **Power Anomalies:** Exceeding inverter limits or unusual charge/discharge asymmetry
- **Efficiency Degradation:** RTE significantly below nominal (85-95% typical)
- **Pattern Deviations:** Unusual departure from expected daily/seasonal patterns

## 16. End of Life Expectations and Timeline Prediction

### 16.1. Research-Validated EOL Timeline
- **80% capacity threshold** typically reached after 5-7 years (significant system variation)
- **First-generation systems** performed better than initial predictions
- **10-year warranties** achievable for most systems with manufacturer degradation reserves

### 16.2. Degradation Reserve Understanding
- **Nameplate vs. Actual Capacity:** Manufacturers state less capacity than systems initially have
- **Buffer Purpose:** Helps meet warranty conditions despite normal degradation
- **User Implication:** Real capacity may exceed nameplate capacity initially

## 17. User Behavior and Installation Impact

### 17.1. Household-Dependent Variations
**Normal Variation Factors:**
- **Same system model** shows different operational patterns in different homes
- **PV generation patterns** heavily influence charging behavior
- **Household consumption** determines discharge timing and rates
- **Control strategy settings** vary between installations

### 17.2. Software Updates and System Evolution
**System Changes Over Time:**
- **Software updates** can modify operational behavior
- **Control strategy refinements** may appear as performance changes
- **Derating behavior modifications** over system lifetime
- **Adaptive BMS responses** to aging and environmental conditions
