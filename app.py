from flask import Flask, render_template, request
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai


# ✅ Load environment variables
load_dotenv(dotenv_path=Path('.') / '.env')

# ✅ Configure Gemini with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Create Gemini model instance
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Initialize Flask app (uses 'templates' folder by default)
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/wireless', methods=['GET', 'POST'])
def wireless():
    if request.method == 'POST':
        input_data = {
            'bandwidth_khz': float(request.form['bandwidth_khz']),
            'quantizer_bits': int(request.form['quantizer_bits']),
            'source_encoder_rate': float(request.form['source_encoder_rate']),
            'channel_encoder_rate': float(request.form['channel_encoder_rate']),
            'interleaver_bits': int(request.form['interleaver_bits']),
            'burst_size': int(request.form['burst_size'])
        }
        results = calculate_wireless_rates(input_data)
        explanation = get_ai_explanation("wireless", input_data, results)
        return render_template('wireless.html', results=results, explanation=explanation, input_data=input_data)
    return render_template('wireless.html')

def calculate_wireless_rates(params):
    results = {}

    # Use original form field names
    BW = params['bandwidth_khz'] * 1e3  # Convert from kHz to Hz
    quantizer_bits = params['quantizer_bits']
    Rs = params['source_encoder_rate']
    Rc = params['channel_encoder_rate']
    interleaver_bits = params['interleaver_bits']
    burst_size = params['burst_size']

    # Step 1: Sampling rate
    fs = 2 * BW
    results['sampler_rate'] = fs

    # Step 2: Quantizer output bit rate
    quantizer_out_bit_rate = quantizer_bits * fs
    results['quantizer_rate'] = quantizer_out_bit_rate

    # Step 3: Source encoder output bit rate
    source_encoder_out_bit_rate = quantizer_out_bit_rate * Rs
    results['source_encoder_rate'] = source_encoder_out_bit_rate

    # Step 4: Channel encoder output bit rate
    channel_encoder_out_bit_rate = source_encoder_out_bit_rate * (1 / Rc)
    results['channel_encoder_rate'] = channel_encoder_out_bit_rate

    # Step 5: Interleaver output bit rate
    interleaver_out_bit_rate = channel_encoder_out_bit_rate
    results['interleaver_rate'] = interleaver_out_bit_rate

    # Step 6: Burst formatting output bit rate
    burst_formatting_rate = interleaver_out_bit_rate / burst_size
    results['burst_formatting_rate'] = burst_formatting_rate

    return results



@app.route('/ofdm', methods=['GET', 'POST'])
def ofdm():
    if request.method == 'POST':
        input_data = {
            'BW_resource_block': float(request.form['BW_resource_block']),
            'subcarrier_spacing': float(request.form['subcarrier_spacing']),
            'num_ofdm_symbols_per_resource_block': int(request.form['num_ofdm_symbols_per_resource_block']),
            'resource_block_duration': float(request.form['resource_block_duration']),
            'num_modulated_bits': int(request.form['num_modulated_bits']),
            'num_of_parallel_resource_blocks': int(request.form['num_of_parallel_resource_blocks'])
        }
        results = calculate_ofdm_rates(input_data)
        explanation = get_ai_explanation("ofdm", input_data, results)
        return render_template('ofdm.html', results=results, explanation=explanation, input_data=input_data)
    return render_template('ofdm.html')


def calculate_ofdm_rates(params):
    from math import log

    results = {}
    
    BW_resource_block = params['BW_resource_block']
    subcarrier_spacing = params['subcarrier_spacing']
    num_symbols = params['num_ofdm_symbols_per_resource_block']
    duration = params['resource_block_duration']
    mod_bits = params['num_modulated_bits']
    parallel_blocks = params['num_of_parallel_resource_blocks']

    # Step 1: Bits per resource element
    num_bits_resource_element = round(log(mod_bits, 2))
    results['num_bits_resource_element'] = num_bits_resource_element

    # Step 2: Bits per OFDM symbol
    num_bits_per_ofdm_symbol = round(num_bits_resource_element * (BW_resource_block / subcarrier_spacing))
    results['num_bits_per_ofdm_symbol'] = num_bits_per_ofdm_symbol

    # Step 3: Bits per resource block
    num_bits_per_resource_block = num_bits_per_ofdm_symbol * num_symbols
    results['num_bits_per_resource_block'] = num_bits_per_resource_block

    # Step 4: Max transmission rate
    max_transmission_rate = round(num_bits_per_resource_block * parallel_blocks / duration)
    results['max_transmission_rate'] = max_transmission_rate

    # Step 5: Spectral efficiency
    total_bandwidth = BW_resource_block * parallel_blocks
    spectral_efficiency = max_transmission_rate / total_bandwidth
    results['spectral_efficiency'] = spectral_efficiency

    return results

@app.route('/link_budget', methods=['GET', 'POST'])
def link_budget():
    if request.method == 'POST':
        input_data = {
            'path_loss_dB': float(request.form['path_loss_dB']),
            'frequency': float(request.form['frequency']),
            'transmitter_antenna_gain_dB': float(request.form['transmitter_antenna_gain_dB']),
            'receiver_antenna_gain_dB': float(request.form['receiver_antenna_gain_dB']),
            'data_rate': float(request.form['data_rate']),
            'feed_line_loss_dB': float(request.form['feed_line_loss_dB']),
            'other_losses_dB': float(request.form['other_losses_dB']),
            'fade_margin_dB': float(request.form['fade_margin_dB']),
            'receiver_amp_gain_dB': float(request.form['receiver_amp_gain_dB']),
            'transmitter_amp_gain_dB': float(request.form['transmitter_amp_gain_dB']),
            'noise_figure_dB': float(request.form['noise_figure_dB']),
            'noise_temp_kelvin': float(request.form['noise_temp_kelvin']),
            'link_margin_dB': float(request.form['link_margin_dB']),
            'SNR_per_bit_dB': float(request.form['SNR_per_bit_dB']),
        }

        results = calculate_link_budget(input_data)
        explanation = get_ai_explanation("link_budget", input_data, results)
        return render_template('link_budget.html', results=results, explanation=explanation, input_data=input_data)
    return render_template('link_budget.html')

def calculate_link_budget(params):
    def to_dB(value):
        return 10 * np.log10(value)

    results = {}
    K_dB = -228.6  # Boltzmann constant in dB

    # Compute power received (dB)
    power_received_dB = (
        params['link_margin_dB']
        + to_dB(params['noise_temp_kelvin'])
        + K_dB
        + params['noise_figure_dB']
        + to_dB(params['data_rate'])
        + params['SNR_per_bit_dB']
    )
    results['power_received_dB'] = power_received_dB

    # Compute power transmitted (dB)
    power_transmitted_dB = (
        power_received_dB
        + params['path_loss_dB']
        + params['feed_line_loss_dB']
        + params['other_losses_dB']
        + params['fade_margin_dB']
        - params['transmitter_antenna_gain_dB']
        - params['receiver_antenna_gain_dB']
        - params['receiver_amp_gain_dB']
        - params['transmitter_amp_gain_dB']
    )
    results['power_transmitted_dB'] = power_transmitted_dB

    return results


@app.route('/cellular', methods=['GET', 'POST'])
def cellular():
    if request.method == 'POST':
        input_data = {
            'time_slots_per_carrier': int(request.form['time_slots_per_carrier']),
            'total_area': float(request.form['total_area']),
            'max_users': int(request.form['max_users']),
            'avg_call_duration_min': float(request.form['avg_call_duration_min']),
            'avg_call_rate_per_user': float(request.form['avg_call_rate_per_user']),
            'grade_of_service': float(request.form['grade_of_service']),
            'sir': float(request.form['sir']),
            'P0': float(request.form['P0']),
            'receiver_sensitivity': float(request.form['receiver_sensitivity']),
            'd0': float(request.form['d0']),
            'path_loss_exponent': float(request.form['path_loss_exponent'])
        }
        results = calculate_cellular_design(input_data)
        explanation = get_ai_explanation("cellular", input_data, results)
        return render_template('cellular.html', results=results, explanation=explanation, input_data=input_data)
    return render_template('cellular.html')


def calculate_cellular_design(params):
    from math import pi, ceil, log10, sqrt
    import pandas as pd

    results = {}

    # ✅ Load Erlang-B table from raw GitHub URL (MUST be public and raw)
    erlang_url = 'https://raw.githubusercontent.com/ToleenHAMED/wireless-network-app/main/Erlang-B-Table.csv'
    df = pd.read_csv(erlang_url)

    # ✅ Fixed cluster size options and interference factor
    cluster_sizes = [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 28]
    NB = 6

    # ✅ Helper: Convert dB to linear
    def _dB_to_linear(db):
        return 10 ** (db / 10)

    # ✅ Step 1: Traffic calculations
    traffic_per_user = (params['avg_call_duration_min'] / 60) * params['avg_call_rate_per_user']
    total_traffic = traffic_per_user * params['max_users']
    results['traffic_per_user'] = round(traffic_per_user, 4)
    results['total_traffic'] = round(total_traffic, 2)

    # ✅ Step 2: Max distance for reliable communication
    max_distance = ((params['receiver_sensitivity'] / _dB_to_linear(params['P0'])) ** (-1 / params['path_loss_exponent'])) * params['d0']
    results['max_distance_reliable'] = round(max_distance, 2)

    # ✅ Step 3: Max cell size (hexagonal cell approximation)
    max_cell_size = (3 * sqrt(3) / 2) * (max_distance ** 2)
    results['max_cell_size'] = round(max_cell_size, 2)

    # ✅ Step 4: Number of cells in the total area
    total_cells = ceil(params['total_area'] / max_cell_size)
    results['total_number_of_cells'] = total_cells

    # ✅ Step 5: Traffic per cell
    traffic_per_cell = total_traffic / total_cells
    results['traffic_per_cell'] = round(traffic_per_cell, 2)

    # ✅ Step 6: Cluster size using SIR
    x = ((_dB_to_linear(params['sir']) * NB) ** (2 / params['path_loss_exponent'])) / 3
    cluster_size = next(N for N in cluster_sizes if N >= x)
    results['cluster_size'] = cluster_size

    # ✅ Step 7: Lookup channels required using Erlang B table and GOS
    column_name = str(int(params['grade_of_service'] * 100)) + '%'
    if column_name not in df.columns:
        raise ValueError(f"Grade of Service {column_name} not found in Erlang-B table.")
    
    try:
        search_value = df[df[column_name] >= traffic_per_cell][column_name].iloc[0]
        channels_required = df[df[column_name] == search_value].iloc[0, 0]
    except IndexError:
        raise ValueError("Traffic value exceeds Erlang-B table range.")
    
    results['channels_required'] = int(channels_required)

    # ✅ Step 8: Calculate carriers
    carriers_per_cell = ceil(channels_required / params['time_slots_per_carrier'])
    results['carriers_per_cell'] = carriers_per_cell
    results['carriers_in_system'] = carriers_per_cell * cluster_size

    return results




def get_ai_explanation(scenario, inputs, results):
    prompt = f"""
You are a wireless communications expert explaining technical concepts to engineering students.

Scenario: {scenario}
Input Parameters: {inputs}
Calculation Results: {results}

Please provide a clear, concise explanation of:
1. What these calculations mean
2. How the inputs relate to the outputs
3. Practical implications of these results
4. Any limitations or assumptions

Use bullet points for clarity and keep the explanation under 200 words.
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"


# ✅ Entry point
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
