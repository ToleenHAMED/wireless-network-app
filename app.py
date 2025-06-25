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
            'sampling_rate': float(request.form['sampling_rate']),
            'bits_per_sample': int(request.form['bits_per_sample']),
            'source_coding_rate': float(request.form['source_coding_rate']),
            'channel_coding_rate': float(request.form['channel_coding_rate']),
            'interleaver_depth': int(request.form['interleaver_depth']),
            'burst_size': int(request.form['burst_size'])
        }
        results = calculate_wireless_rates(input_data)
        explanation = get_ai_explanation("wireless", input_data, results)
        return render_template('wireless.html', results=results, explanation=explanation, input_data=input_data)
    return render_template('wireless.html')


def calculate_wireless_rates(params):
    results = {}
    results['sampler_rate'] = params['sampling_rate']
    results['quantizer_rate'] = params['sampling_rate'] * params['bits_per_sample']
    results['source_encoder_rate'] = results['quantizer_rate'] * params['source_coding_rate']
    results['channel_encoder_rate'] = results['source_encoder_rate'] / params['channel_coding_rate']
    results['interleaver_rate'] = results['channel_encoder_rate']
    results['burst_formatting_rate'] = results['channel_encoder_rate'] / params['burst_size']
    return results


@app.route('/ofdm', methods=['GET', 'POST'])
def ofdm():
    if request.method == 'POST':
        input_data = {
            'subcarriers': int(request.form['subcarriers']),
            'subcarrier_spacing': float(request.form['subcarrier_spacing']),
            'symbols_per_slot': int(request.form['symbols_per_slot']),
            'slots_per_frame': int(request.form['slots_per_frame']),
            'bits_per_symbol': int(request.form['bits_per_symbol']),
            'resource_blocks': int(request.form['resource_blocks'])
        }
        results = calculate_ofdm_rates(input_data)
        explanation = get_ai_explanation("ofdm", input_data, results)
        return render_template('ofdm.html', results=results, explanation=explanation, input_data=input_data)
    return render_template('ofdm.html')


def calculate_ofdm_rates(params):
    results = {}
    results['resource_element_rate'] = params['bits_per_symbol'] * params['subcarrier_spacing']
    results['ofdm_symbol_rate'] = results['resource_element_rate'] * params['subcarriers']
    results['resource_block_rate'] = results['ofdm_symbol_rate'] * params['symbols_per_slot']
    results['max_transmission_capacity'] = results['resource_block_rate'] * params['resource_blocks'] * params['slots_per_frame']
    total_bandwidth = params['subcarriers'] * params['subcarrier_spacing']
    results['spectral_efficiency'] = results['max_transmission_capacity'] / total_bandwidth
    return results


@app.route('/link_budget', methods=['GET', 'POST'])
def link_budget():
    if request.method == 'POST':
        input_data = {
            'transmitter_power': float(request.form['transmitter_power']),
            'transmitter_gain': float(request.form['transmitter_gain']),
            'receiver_gain': float(request.form['receiver_gain']),
            'frequency': float(request.form['frequency']),
            'distance': float(request.form['distance']),
            'system_losses': float(request.form['system_losses'])
        }
        results = calculate_link_budget(input_data)
        explanation = get_ai_explanation("link_budget", input_data, results)
        return render_template('link_budget.html', results=results, explanation=explanation, input_data=input_data)
    return render_template('link_budget.html')


def calculate_link_budget(params):
    results = {}
    speed_of_light = 3e8
    fspl = 20 * np.log10(params['distance']) + 20 * np.log10(params['frequency']) + 20 * np.log10(4 * np.pi / speed_of_light)
    results['path_loss'] = fspl
    eirp = params['transmitter_power'] + params['transmitter_gain']
    results['eirp'] = eirp
    rss = eirp + params['receiver_gain'] - fspl - params['system_losses']
    results['received_signal_strength'] = rss
    return results


@app.route('/cellular', methods=['GET', 'POST'])
def cellular():
    if request.method == 'POST':
        input_data = {
            'area_size': float(request.form['area_size']),
            'subscriber_density': float(request.form['subscriber_density']),
            'traffic_per_user': float(request.form['traffic_per_user']),
            'cell_radius': float(request.form['cell_radius']),
            'frequency_reuse_factor': int(request.form['frequency_reuse_factor']),
            'sectorization': request.form['sectorization']
        }
        results = calculate_cellular_design(input_data)
        explanation = get_ai_explanation("cellular", input_data, results)
        return render_template('cellular.html', results=results, explanation=explanation, input_data=input_data)
    return render_template('cellular.html')


def calculate_cellular_design(params):
    results = {}
    total_traffic = params['area_size'] * params['subscriber_density'] * params['traffic_per_user']
    results['total_traffic'] = total_traffic
    cell_area = np.pi * params['cell_radius']**2
    results['cell_area'] = cell_area
    num_cells = np.ceil(params['area_size'] / cell_area)
    results['num_cells'] = int(num_cells)
    traffic_per_cell = total_traffic / num_cells
    results['traffic_per_cell'] = traffic_per_cell
    sector_factor = {'none': 1, '3-sector': 3, '6-sector': 6}.get(params['sectorization'], 1)
    results['channels_per_cell'] = np.ceil(traffic_per_cell * sector_factor)
    results['system_capacity'] = results['channels_per_cell'] * num_cells
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
