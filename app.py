from flask import Flask, render_template, request
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai

# ✅ Load environment variables
load_dotenv(dotenv_path=Path('.') / '.env')
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')

# ✅ Embedded Erlang-B Table (partial view for brevity; extend if needed)
ERLANG_B_TABLE = {
    "(N)": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150],
    "0.1%": [0.001, 0.046, 0.194, 0.439, 0.762, 1.150, 1.580, 2.050, 2.560, 3.090, 3.650, 4.230, 4.830, 5.450, 6.080, 6.720, 7.380, 8.050, 8.720, 9.410, 10.100, 10.800, 11.500, 12.200, 13.000, 13.700, 14.400, 15.200, 15.900, 16.700, 17.400, 18.200, 19.000, 19.700, 20.500, 21.300, 22.100, 22.900, 23.700, 24.400, 25.200, 26.000, 26.800, 27.600, 28.400, 29.300, 30.100, 30.900, 31.700, 32.500, 33.300, 34.200, 35.000, 35.800, 36.600, 37.500, 38.300, 39.100, 40.000, 40.800, 41.600, 42.500, 43.300, 44.200, 45.000, 45.800, 46.700, 47.500, 48.400, 49.200, 50.100, 50.900, 51.800, 52.700, 53.500, 54.400, 55.200, 56.100, 56.900, 57.800, 58.700, 59.500, 60.400, 61.300, 62.100, 63.000, 63.900, 64.700, 65.600, 66.500, 67.400, 68.200, 69.100, 70.000, 70.900, 71.700, 72.600, 73.500, 74.400, 75.200, 76.100, 77.000, 77.900, 78.800, 79.600, 80.500, 81.400, 82.300, 83.200, 84.100, 85.000, 85.800, 86.700, 87.600, 88.500, 89.400, 90.300, 91.200, 92.100, 93.000, 93.900, 94.700, 95.600, 96.500, 97.400, 98.300, 99.200, 100.100, 101.000, 101.900, 102.800, 103.700, 104.600, 105.500, 106.400, 107.300, 108.200, 109.100, 110.000, 110.900, 111.800, 112.700, 113.500, 114.400, 115.300, 116.300, 117.200, 118.100, 119.000, 119.900],
    "0.2%": [0.002, 0.065, 0.249, 0.535, 0.900, 1.330, 1.800, 2.310, 2.850, 3.430, 4.020, 4.640, 5.270, 5.920, 6.580, 7.260, 7.950, 8.640, 9.350, 10.100, 10.800, 11.500, 12.300, 13.000, 13.800, 14.500, 15.300, 16.100, 16.800, 17.600, 18.400, 19.200, 20.000, 20.800, 21.600, 22.400, 23.200, 24.000, 24.800, 25.600, 26.400, 27.200, 28.100, 28.900, 29.700, 30.500, 31.400, 32.200, 33.000, 33.900, 34.700, 35.600, 36.400, 37.200, 38.100, 38.900, 39.800, 40.600, 41.500, 42.400, 43.200, 44.100, 44.900, 45.800, 46.600, 47.500, 48.400, 49.200, 50.100, 51.000, 51.800, 52.700, 53.600, 54.500, 55.300, 56.200, 57.100, 58.000, 58.800, 59.700, 60.600, 61.500, 62.400, 63.200, 64.100, 65.000, 65.900, 66.800, 67.700, 68.600, 69.400, 70.300, 71.200, 72.100, 73.000, 73.900, 74.800, 75.700, 76.600, 77.500, 78.400, 79.300, 80.200, 81.100, 82.000, 82.800, 83.700, 84.600, 85.500, 86.400, 87.300, 88.300, 89.200, 90.100, 91.000, 91.900, 92.800, 93.700, 94.600, 95.500, 96.400, 97.300, 98.200, 99.100, 100.000, 100.900, 101.800, 102.800, 103.700, 104.600, 105.500, 106.400, 107.300, 108.200, 109.100, 110.000, 111.000, 111.900, 112.800, 113.700, 114.600, 115.500, 116.300, 117.200, 118.200, 119.200, 120.100, 121.000, 121.900, 122.800],
    "0.5%": [0.005, 0.105, 0.349, 0.701, 1.130, 1.620, 2.160, 2.730, 3.330, 3.960, 4.610, 5.280, 5.960, 6.660, 7.380, 8.100, 8.830, 9.580, 10.300, 11.100, 11.900, 12.600, 13.400, 14.200, 15.000, 15.800, 16.600, 17.400, 18.200, 19.000, 19.900, 20.700, 21.500, 22.300, 23.200, 24.000, 24.800, 25.700, 26.500, 27.400, 28.200, 29.100, 29.900, 30.800, 31.700, 32.500, 33.400, 34.200, 35.100, 36.000, 36.900, 37.700, 38.600, 39.500, 40.400, 41.200, 42.100, 43.000, 43.900, 44.800, 45.600, 46.500, 47.400, 48.300, 49.200, 50.100, 51.000, 51.900, 52.800, 53.700, 54.600, 55.500, 56.400, 57.300, 58.200, 59.100, 60.000, 60.900, 61.800, 62.700, 63.600, 64.500, 65.400, 66.300, 67.200, 68.100, 69.000, 69.900, 70.800, 71.800, 72.700, 73.600, 74.500, 75.400, 76.300, 77.200, 78.200, 79.100, 80.000, 80.900, 81.800, 82.700, 83.700, 84.600, 85.500, 86.400, 87.400, 88.300, 89.200, 90.100, 91.000, 92.000, 92.900, 93.800, 94.700, 95.700, 96.600, 97.500, 98.500, 99.400, 100.300, 101.200, 102.200, 103.100, 104.000, 105.000, 105.900, 106.800, 107.800, 108.700, 109.600, 110.600, 111.500, 112.400, 113.300, 114.300, 115.200, 116.200, 117.100, 118.000, 118.900, 120.000, 120.800, 121.800, 122.700, 123.700, 124.600, 125.500, 126.400, 127.400],
    "1%": [0.010, 0.153, 0.455, 0.869, 1.360, 1.910, 2.500, 3.130, 3.780, 4.460, 5.160, 5.880, 6.610, 7.350, 8.110, 8.880, 9.650, 10.400, 11.200, 12.000, 12.800, 13.700, 14.500, 15.300, 16.100, 17.000, 17.800, 18.600, 19.500, 20.300, 21.200, 22.000, 22.900, 23.800, 24.600, 25.500, 26.400, 27.300, 28.100, 29.000, 29.900, 30.800, 31.700, 32.500, 33.400, 34.300, 35.200, 36.100, 37.000, 37.900, 38.800, 39.700, 40.600, 41.500, 42.400, 43.300, 44.200, 45.100, 46.000, 46.900, 47.900, 48.800, 49.700, 50.600, 51.500, 52.400, 53.400, 54.300, 55.200, 56.100, 57.000, 58.000, 58.900, 59.800, 60.700, 61.700, 62.600, 63.500, 64.400, 65.400, 66.300, 67.200, 68.200, 69.100, 70.000, 70.900, 71.900, 72.800, 73.700, 74.700, 75.600, 76.600, 77.500, 78.400, 79.400, 80.300, 81.200, 82.200, 83.100, 84.100, 85.000, 85.900, 86.900, 87.800, 88.800, 89.700, 90.700, 91.600, 92.500, 93.500, 94.400, 95.400, 96.300, 97.300, 98.200, 99.200, 100.100, 101.100, 102.000, 103.000, 103.900, 104.900, 105.800, 106.800, 107.700, 108.700, 109.600, 110.600, 111.500, 112.500, 113.400, 114.400, 115.300, 116.300, 117.200, 118.200, 119.100, 120.100, 121.000, 122.000, 123.000, 123.900, 124.900, 125.900, 126.800, 127.800, 128.700, 129.600, 130.600, 131.600],
    "1.2%": [0.012, 0.168, 0.489, 0.922, 1.430, 2.000, 2.600, 3.250, 3.920, 4.610, 5.320, 6.050, 6.800, 7.560, 8.330, 9.110, 9.890, 10.700, 11.500, 12.300, 13.100, 14.000, 14.800, 15.600, 16.500, 17.300, 18.200, 19.000, 19.900, 20.700, 21.600, 22.500, 23.300, 24.200, 25.100, 26.000, 26.800, 27.700, 28.600, 29.500, 30.400, 31.300, 32.200, 33.100, 34.000, 34.900, 35.800, 36.700, 37.600, 38.500, 39.400, 40.300, 41.200, 42.100, 43.000, 43.900, 44.800, 45.800, 46.700, 47.600, 48.500, 49.400, 50.400, 51.300, 52.200, 53.100, 54.100, 55.000, 55.900, 56.800, 57.800, 58.700, 59.600, 60.600, 61.500, 62.400, 63.400, 64.300, 65.200, 66.200, 67.100, 68.000, 69.000, 69.900, 70.900, 71.800, 72.700, 73.700, 74.600, 75.600, 76.500, 77.400, 78.400, 79.300, 80.300, 81.200, 82.200, 83.100, 84.100, 85.000, 86.000, 86.900, 87.800, 88.800, 89.700, 90.700, 91.600, 92.600, 93.500, 94.500, 95.500, 96.400, 97.400, 98.300, 99.300, 100.200, 101.200, 102.100, 103.100, 104.000, 105.000, 105.900, 106.900, 107.900, 108.800, 109.800, 110.700, 111.700, 112.600, 113.600, 114.600, 115.500, 116.500, 117.400, 118.400, 119.400, 120.300, 121.300, 122.200, 123.200, 124.200, 125.100, 126.100, 127.100, 128.000, 129.000, 129.900, 130.900, 131.800, 132.800],
    "1.3%": [0.013, 0.176, 0.505, 0.946, 1.460, 2.040, 2.650, 3.300, 3.980, 4.680, 5.400, 6.140, 6.890, 7.650, 8.430, 9.210, 10.000, 10.800, 11.600, 12.400, 13.300, 14.100, 14.900, 15.800, 16.600, 17.500, 18.300, 19.200, 20.000, 20.900, 21.800, 22.600, 23.500, 24.400, 25.300, 26.200, 27.000, 27.900, 28.800, 29.700, 30.600, 31.500, 32.400, 33.300, 34.200, 35.100, 36.000, 36.900, 37.800, 38.700, 39.600, 40.600, 41.500, 42.400, 43.300, 44.200, 45.100, 46.100, 47.000, 47.900, 48.800, 49.700, 50.700, 51.600, 52.500, 53.500, 54.400, 55.300, 56.200, 57.200, 58.100, 59.000, 60.000, 60.900, 61.800, 62.800, 63.700, 64.700, 65.600, 66.500, 67.500, 68.400, 69.400, 70.300, 71.200, 72.200, 73.100, 74.100, 75.000, 76.000, 76.900, 77.800, 78.800, 79.700, 80.700, 81.600, 82.600, 83.500, 84.500, 85.400, 86.400, 87.300, 88.300, 89.200, 90.200, 91.100, 92.100, 93.100, 94.000, 95.000, 95.900, 96.900, 97.800, 98.800, 99.700, 100.700, 101.700, 102.600, 103.600, 104.500, 105.500, 106.400, 107.400, 108.400, 109.300, 110.300, 111.200, 112.200, 113.200, 114.100, 115.100, 116.000, 117.000, 118.000, 118.900, 119.900, 120.900, 121.800, 122.800, 123.700, 124.700, 125.700, 126.700, 127.600, 128.600, 129.500, 130.500, 131.400, 132.500, 133.400],
    "1.5%": [0.020, 0.190, 0.530, 0.990, 1.520, 2.110, 2.730, 3.400, 4.080, 4.800, 5.530, 6.270, 7.030, 7.810, 8.590, 9.390, 10.190, 11.000, 11.820, 12.650, 13.480, 14.320, 15.160, 16.010, 16.870, 17.720, 18.590, 19.450, 20.320, 21.190, 22.070, 22.950, 23.830, 24.720, 25.600, 26.490, 27.390, 28.280, 29.180, 30.080, 30.980, 31.880, 32.790, 33.690, 34.600, 35.510, 36.420, 37.340, 38.250, 39.170, 40.080, 41.000, 41.920, 42.840, 43.770, 44.690, 45.620, 46.540, 47.470, 48.400, 49.330, 50.260, 51.190, 52.120, 53.050, 53.990, 54.920, 55.860, 56.790, 57.730, 58.670, 59.610, 60.550, 61.490, 62.430, 63.370, 64.320, 65.260, 66.200, 67.150, 68.090, 69.040, 69.990, 70.930, 71.880, 72.830, 73.780, 74.730, 75.680, 76.630, 77.580, 78.530, 79.480, 80.430, 81.390, 82.340, 83.290, 84.250, 85.200, 86.160, 87.120, 88.070, 89.030, 89.990, 90.940, 91.900, 92.860, 93.820, 94.780, 95.740, 96.700, 97.660, 98.620, 99.580, 100.540, 101.500, 102.460, 103.430, 104.390, 105.350, 106.310, 107.280, 108.240, 109.210, 110.170, 111.140, 112.100, 113.070, 114.030, 115.000, 115.960, 116.930, 117.900, 118.870, 119.830, 120.800, 121.770, 122.740, 123.710, 124.670, 125.640, 126.660, 127.590, 128.520, 129.540, 130.460, 131.440, 132.380, 133.400, 134.390],
    "2%": [0.020, 0.223, 0.602, 1.090, 1.660, 2.280, 2.940, 3.630, 4.340, 5.080, 5.840, 6.610, 7.400, 8.200, 9.010, 9.830, 10.700, 11.500, 12.300, 13.200, 14.000, 14.900, 15.800, 16.600, 17.500, 18.400, 19.300, 20.200, 21.000, 21.900, 22.800, 23.700, 24.600, 25.500, 26.400, 27.300, 28.300, 29.200, 30.100, 31.000, 31.900, 32.800, 33.800, 34.700, 35.600, 36.500, 37.500, 38.400, 39.300, 40.300, 41.200, 42.100, 43.100, 44.000, 44.900, 45.900, 46.800, 47.800, 48.700, 49.600, 50.600, 51.500, 52.500, 53.400, 54.400, 55.300, 56.300, 57.200, 58.200, 59.100, 60.100, 61.000, 62.000, 62.900, 63.900, 64.900, 65.800, 66.800, 67.700, 68.700, 69.600, 70.600, 71.600, 72.500, 73.500, 74.500, 75.400, 76.400, 77.300, 78.300, 79.300, 80.200, 81.200, 82.200, 83.100, 84.100, 85.100, 86.000, 87.000, 88.000, 88.900, 89.900, 90.900, 91.900, 92.800, 93.800, 94.800, 95.700, 96.700, 97.700, 98.700, 99.600, 100.600, 101.600, 102.500, 103.500, 104.500, 105.500, 106.400, 107.400, 108.400, 109.400, 110.300, 111.300, 112.300, 113.300, 114.300, 115.200, 116.200, 117.200, 118.200, 119.100, 120.100, 121.100, 122.100, 123.100, 124.000, 125.000, 126.000, 127.000, 128.000, 129.000, 129.900, 130.900, 132.000, 132.900, 133.900, 134.800, 135.800, 136.800],
    "3%": [0.031, 0.282, 0.715, 1.260, 1.880, 2.540, 3.250, 3.990, 4.750, 5.530, 6.330, 7.140, 7.970, 8.800, 9.650, 10.500, 11.400, 12.200, 13.100, 14.000, 14.900, 15.800, 16.700, 17.600, 18.500, 19.400, 20.300, 21.200, 22.100, 23.100, 24.000, 24.900, 25.800, 26.800, 27.700, 28.600, 29.600, 30.500, 31.500, 32.400, 33.400, 34.300, 35.300, 36.200, 37.200, 38.100, 39.100, 40.000, 41.000, 41.900, 42.900, 43.900, 44.800, 45.800, 46.700, 47.700, 48.700, 49.600, 50.600, 51.600, 52.500, 53.500, 54.500, 55.400, 56.400, 57.400, 58.400, 59.300, 60.300, 61.300, 62.300, 63.200, 64.200, 65.200, 66.200, 67.200, 68.100, 69.100, 70.100, 71.100, 72.100, 73.000, 74.000, 75.000, 76.000, 77.000, 78.000, 78.900, 79.900, 80.900, 81.900, 82.900, 83.900, 84.900, 85.800, 86.800, 87.800, 88.800, 89.800, 90.800, 91.800, 92.800, 93.800, 94.800, 95.700, 96.700, 97.700, 98.700, 99.700, 100.700, 101.700, 102.700, 103.700, 104.700, 105.700, 106.700, 107.700, 108.700, 109.700, 110.700, 111.600, 112.600, 113.600, 114.600, 115.600, 116.600, 117.600, 118.600, 119.600, 120.600, 121.600, 122.600, 123.600, 124.600, 125.600, 126.600, 127.600, 128.600, 129.600, 130.600, 131.600, 132.600, 133.600, 134.600, 135.700, 136.600, 137.600, 138.600, 139.600, 140.700],
    "5%": [0.053, 0.381, 0.899, 1.52, 2.22, 2.96, 3.74, 4.54, 5.37, 6.22, 7.08, 7.95, 8.83, 9.73, 10.6, 11.5, 12.5, 13.4, 14.3, 15.2, 16.2, 17.1, 18.1, 19.0, 20.0, 20.9, 21.9, 22.9, 23.8, 24.8, 25.8, 26.7, 27.7, 28.7, 29.7, 30.7, 31.6, 32.6, 33.6, 34.6, 35.6, 36.6, 37.6, 38.6, 39.6, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5, 47.5, 48.5, 49.5, 50.5, 51.5, 52.6, 53.6, 54.6, 55.6, 56.6, 57.6, 58.6, 59.6, 60.6, 61.6, 62.6, 63.7, 64.7, 65.7, 66.7, 67.7, 68.7, 69.7, 70.8, 71.8, 72.8, 73.8, 74.8, 75.8, 76.9, 77.9, 78.9, 79.9, 80.9, 82.0, 83.0, 84.0, 85.0, 86.0, 87.1, 88.1, 89.1, 90.1, 91.1, 92.2, 93.2, 94.2, 95.2, 96.3, 97.3, 98.3, 99.3, 100.4, 101.4, 102.4, 103.4, 104.5, 105.5, 106.5, 107.5, 108.6, 109.6, 110.6, 111.7, 112.7, 113.7, 114.7, 115.8, 116.8, 117.8, 118.9, 119.9, 120.9, 121.9, 123.0, 124.0, 125.0, 126.1, 127.1, 128.1, 129.2, 130.2, 131.2, 132.2, 133.3, 134.3, 135.3, 136.4, 137.4, 138.5, 139.5, 140.5, 141.6, 142.6, 143.6, 144.7, 145.7, 146.7],
    "7%": [0.075, 0.470, 1.060, 1.750, 2.500, 3.300, 4.140, 5.000, 5.880, 6.780, 7.690, 8.610, 9.540, 10.500, 11.400, 12.400, 13.400, 14.300, 15.300, 16.300, 17.300, 18.200, 19.200, 20.200, 21.200, 22.200, 23.200, 24.200, 25.200, 26.200, 27.200, 28.200, 29.300, 30.300, 31.300, 32.300, 33.300, 34.400, 35.400, 36.400, 37.400, 38.400, 39.500, 40.500, 41.500, 42.600, 43.600, 44.600, 45.700, 46.700, 47.700, 48.800, 49.800, 50.800, 51.900, 52.900, 53.900, 55.000, 56.000, 57.100, 58.100, 59.100, 60.200, 61.200, 62.300, 63.300, 64.400, 65.400, 66.400, 67.500, 68.500, 69.600, 70.600, 71.700, 72.700, 73.800, 74.800, 75.900, 76.900, 78.000, 79.000, 80.100, 81.100, 82.200, 83.200, 84.300, 85.300, 86.400, 87.400, 88.500, 89.500, 90.600, 91.600, 92.700, 93.700, 94.800, 95.800, 96.900, 97.900, 99.000, 100.000, 101.100, 102.200, 103.200, 104.300, 105.300, 106.400, 107.400, 108.500, 109.500, 110.600, 111.700, 112.700, 113.800, 114.800, 115.900, 116.900, 118.000, 119.100, 120.100, 121.200, 122.200, 123.300, 124.400, 125.400, 126.500, 127.500, 128.600, 129.600, 130.700, 131.800, 132.800, 133.900, 134.900, 136.000, 137.100, 138.100, 139.200, 140.200, 141.300, 142.400, 143.500, 144.500, 145.500, 146.600, 147.600, 148.800, 149.800, 150.800, 151.900],
    "10%": [0.111, 0.595, 1.270, 2.050, 2.880, 3.760, 4.670, 5.600, 6.550, 7.510, 8.490, 9.470, 10.500, 11.500, 12.500, 13.500, 14.500, 15.500, 16.600, 17.600, 18.700, 19.700, 20.700, 21.800, 22.800, 23.900, 24.900, 26.000, 27.100, 28.100, 29.200, 30.200, 31.300, 32.400, 33.400, 34.500, 35.600, 36.600, 37.700, 38.800, 39.900, 40.900, 42.000, 43.100, 44.200, 45.200, 46.300, 47.400, 48.500, 49.600, 50.600, 51.700, 52.800, 53.900, 55.000, 56.100, 57.100, 58.200, 59.300, 60.400, 61.500, 62.600, 63.700, 64.800, 65.800, 66.900, 68.000, 69.100, 70.200, 71.300, 72.400, 73.500, 74.600, 75.600, 76.700, 77.800, 78.900, 80.000, 81.100, 82.200, 83.300, 84.400, 85.500, 86.600, 87.700, 88.800, 89.900, 91.000, 92.100, 93.100, 94.200, 95.300, 96.400, 97.500, 98.600, 99.700, 100.800, 101.900, 103.000, 104.100, 105.200, 106.300, 107.400, 108.500, 109.600, 110.700, 111.800, 112.900, 114.000, 115.100, 116.200, 117.300, 118.400, 119.500, 120.600, 121.700, 122.800, 123.900, 125.000, 126.100, 127.200, 128.300, 129.400, 130.500, 131.600, 132.700, 133.800, 134.900, 136.000, 137.100, 138.200, 139.300, 140.400, 141.500, 142.600, 143.700, 144.800, 145.900, 147.000, 148.100, 149.200, 150.300, 151.400, 152.500, 153.600, 154.700, 155.800, 156.900, 158.000, 159.100],
    "15%": [0.176, 0.796, 1.600, 2.500, 3.450, 4.440, 5.460, 6.500, 7.550, 8.620, 9.690, 10.800, 11.900, 13.000, 14.100, 15.200, 16.300, 17.400, 18.500, 19.600, 20.800, 21.900, 23.000, 24.200, 25.300, 26.400, 27.600, 28.700, 29.900, 31.000, 32.100, 33.300, 34.400, 35.600, 36.700, 37.900, 39.000, 40.200, 41.300, 42.500, 43.600, 44.800, 45.900, 47.100, 48.200, 49.400, 50.600, 51.700, 52.900, 54.000, 55.200, 56.300, 57.500, 58.700, 59.800, 61.000, 62.100, 63.300, 64.500, 65.600, 66.800, 68.000, 69.100, 70.300, 71.400, 72.600, 73.800, 74.900, 76.100, 77.300, 78.400, 79.600, 80.800, 81.900, 83.100, 84.200, 85.400, 86.600, 87.700, 88.900, 90.100, 91.200, 92.400, 93.600, 94.700, 95.900, 97.100, 98.200, 99.400, 100.600, 101.700, 102.900, 104.100, 105.300, 106.400, 107.600, 108.800, 109.900, 111.100, 112.300, 113.400, 114.600, 115.800, 116.900, 118.100, 119.300, 120.400, 121.600, 122.800, 124.000, 125.100, 126.300, 127.500, 128.600, 129.800, 131.000, 132.100, 133.300, 134.500, 135.700, 136.800, 138.000, 139.200, 140.300, 141.500, 142.700, 143.900, 145.000, 146.200, 147.400, 148.500, 149.700, 150.900, 152.000, 153.200, 154.400, 155.600, 156.700, 157.900, 159.100, 160.200, 161.400, 162.600, 163.800, 164.900, 166.100, 167.300, 168.500, 169.600, 170.800],
    "20%": [0.250, 1.000, 1.930, 2.950, 4.010, 5.110, 6.230, 7.370, 8.520, 9.680, 10.900, 12.000, 13.200, 14.400, 15.600, 16.800, 18.000, 19.200, 20.400, 21.600, 22.800, 24.100, 25.300, 26.500, 27.700, 28.900, 30.200, 31.400, 32.600, 33.800, 35.100, 36.300, 37.500, 38.800, 40.000, 41.200, 42.400, 43.700, 44.900, 46.100, 47.400, 48.600, 49.900, 51.100, 52.300, 53.600, 54.800, 56.000, 57.300, 58.500, 59.700, 61.000, 62.200, 63.500, 64.700, 65.900, 67.200, 68.400, 69.700, 70.900, 72.100, 73.400, 74.600, 75.900, 77.100, 78.300, 79.600, 80.800, 82.100, 83.300, 84.600, 85.800, 87.000, 88.300, 89.500, 90.800, 92.000, 93.300, 94.500, 95.700, 97.000, 98.200, 99.500, 100.700, 102.000, 103.200, 104.500, 105.700, 106.900, 108.200, 109.400, 110.700, 111.900, 113.200, 114.400, 115.700, 116.900, 118.200, 119.400, 120.600, 121.900, 123.100, 124.400, 125.600, 126.900, 128.100, 129.400, 130.600, 131.900, 133.100, 134.300, 135.600, 136.800, 138.100, 139.300, 140.600, 141.800, 143.100, 144.300, 145.600, 146.800, 148.100, 149.300, 150.600, 151.800, 153.000, 154.300, 155.500, 156.800, 158.000, 159.300, 160.500, 161.800, 163.000, 164.300, 165.500, 166.800, 168.000, 169.300, 170.500, 171.800, 173.000, 174.200, 175.500, 176.700, 178.000, 179.200, 180.500, 181.700, 183.000],
    "30%": [0.429, 1.450, 2.630, 3.890, 5.190, 6.510, 7.860, 9.210, 10.600, 12.000, 13.300, 14.700, 16.100, 17.500, 18.900, 20.300, 21.700, 23.100, 24.500, 25.900, 27.300, 28.700, 30.100, 31.600, 33.000, 34.400, 35.800, 37.200, 38.600, 40.000, 41.500, 42.900, 44.300, 45.700, 47.100, 48.600, 50.000, 51.400, 52.800, 54.200, 55.700, 57.100, 58.500, 59.900, 61.300, 62.800, 64.200, 65.600, 67.000, 68.500, 69.900, 71.300, 72.700, 74.200, 75.600, 77.000, 78.400, 79.800, 81.300, 82.700, 84.100, 85.500, 87.000, 88.400, 89.800, 91.200, 92.700, 94.100, 95.500, 96.900, 98.400, 99.800, 101.200, 102.700, 104.100, 105.500, 106.900, 108.400, 109.800, 111.200, 112.600, 114.100, 115.500, 116.900, 118.300, 119.800, 121.200, 122.600, 124.000, 125.500, 126.900, 128.300, 129.700, 131.200, 132.600, 134.000, 135.500, 136.900, 138.300, 139.700, 141.200, 142.600, 144.000, 145.400, 146.900, 148.300, 149.700, 151.100, 152.600, 154.000, 155.400, 156.900, 158.300, 159.700, 161.100, 162.600, 164.000, 165.400, 166.800, 168.300, 169.700, 171.100, 172.600, 174.000, 175.400, 176.800, 178.300, 179.700, 181.100, 182.500, 184.000, 185.400, 186.800, 188.300, 189.700, 191.100, 192.500, 194.000, 195.400, 196.800, 198.300, 199.700, 201.100, 202.500, 204.000, 205.400, 206.800, 208.200, 209.700, 211.100],
    "40%": [0.667, 2.000, 3.480, 5.020, 6.600, 8.190, 9.800, 11.400, 13.000, 14.700, 16.300, 18.000, 19.600, 21.200, 22.900, 24.500, 26.200, 27.800, 29.500, 31.200, 32.800, 34.500, 36.100, 37.800, 39.400, 41.100, 42.800, 44.400, 46.100, 47.700, 49.400, 51.100, 52.700, 54.400, 56.000, 57.700, 59.400, 61.000, 62.700, 64.400, 66.000, 67.700, 69.300, 71.000, 72.700, 74.300, 76.000, 77.700, 79.300, 81.000, 82.700, 84.300, 86.000, 87.600, 89.300, 91.000, 92.600, 94.300, 96.000, 97.600, 99.300, 101.000, 102.600, 104.300, 106.000, 107.600, 109.300, 111.000, 112.600, 114.300, 115.900, 117.600, 119.300, 120.900, 122.600, 124.300, 125.900, 127.600, 129.300, 130.900, 132.600, 134.300, 135.900, 137.600, 139.300, 140.900, 142.600, 144.300, 145.900, 147.600, 149.300, 150.900, 152.600, 154.300, 155.900, 157.600, 159.300, 160.900, 162.600, 164.300, 165.900, 167.600, 169.200, 170.900, 172.600, 174.200, 175.900, 177.600, 179.200, 180.900, 182.600, 184.200, 185.900, 187.600, 189.200, 190.900, 192.600, 194.200, 195.900, 197.600, 199.200, 200.900, 202.600, 204.200, 205.900, 207.600, 209.200, 210.900, 212.600, 214.200, 215.900, 217.600, 219.200, 220.900, 222.600, 224.200, 225.900, 227.600, 229.200, 230.900, 232.600, 234.200, 235.900, 237.600, 239.200, 240.900, 242.600, 244.200, 245.900, 247.600],
    "50%": [1.000, 2.730, 4.590, 6.500, 8.440, 10.400, 12.400, 14.300, 16.300, 18.300, 20.300, 22.200, 24.200, 26.200, 28.200, 30.200, 32.200, 34.200, 36.200, 38.200, 40.200, 42.100, 44.100, 46.100, 48.100, 50.100, 52.100, 54.100, 56.100, 58.100, 60.100, 62.100, 64.100, 66.100, 68.100, 70.100, 72.100, 74.100, 76.100, 78.100, 80.100, 82.100, 84.100, 86.100, 88.100, 90.100, 92.100, 94.100, 96.100, 98.100, 100.100, 102.100, 104.100, 106.100, 108.100, 110.100, 112.100, 114.100, 116.100, 118.100, 120.100, 122.100, 124.100, 126.100, 128.100, 130.100, 132.100, 134.100, 136.100, 138.100, 140.100, 142.100, 144.100, 146.100, 148.000, 150.000, 152.000, 154.000, 156.000, 158.000, 160.000, 162.000, 164.000, 166.000, 168.000, 170.000, 172.000, 174.000, 176.000, 178.000, 180.000, 182.000, 184.000, 186.000, 188.000, 190.000, 192.000, 194.000, 196.000, 198.000, 200.000, 202.000, 204.000, 206.000, 208.000, 210.000, 212.000, 214.000, 216.000, 218.000, 220.000, 222.000, 224.000, 226.000, 228.000, 230.000, 232.000, 234.000, 236.000, 238.000, 240.000, 242.000, 244.000, 246.000, 248.000, 250.000, 252.000, 254.000, 256.000, 258.000, 260.000, 262.000, 264.000, 266.000, 268.000, 270.000, 272.000, 274.000, 276.000, 278.000, 280.000, 282.000, 284.000, 286.000, 288.000, 290.000, 292.000, 294.000, 296.000, 298.000],
}

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
    BW = params['bandwidth_khz'] * 1e3
    quantizer_bits = params['quantizer_bits']
    Rs = params['source_encoder_rate']
    Rc = params['channel_encoder_rate']
    interleaver_bits = params['interleaver_bits']
    burst_size = params['burst_size']

    fs = 2 * BW
    results['sampler_rate'] = fs

    quantizer_out_bit_rate = quantizer_bits * fs
    results['quantizer_rate'] = quantizer_out_bit_rate

    source_encoder_out_bit_rate = quantizer_out_bit_rate * Rs
    results['source_encoder_rate'] = source_encoder_out_bit_rate

    channel_encoder_out_bit_rate = source_encoder_out_bit_rate * (1 / Rc)
    results['channel_encoder_rate'] = channel_encoder_out_bit_rate

    interleaver_out_bit_rate = channel_encoder_out_bit_rate
    results['interleaver_rate'] = interleaver_out_bit_rate

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

    num_bits_resource_element = round(log(mod_bits, 2))
    results['num_bits_resource_element'] = num_bits_resource_element

    num_bits_per_ofdm_symbol = round(num_bits_resource_element * (BW_resource_block / subcarrier_spacing))
    results['num_bits_per_ofdm_symbol'] = num_bits_per_ofdm_symbol

    num_bits_per_resource_block = num_bits_per_ofdm_symbol * num_symbols
    results['num_bits_per_resource_block'] = num_bits_per_resource_block

    max_transmission_rate = round(num_bits_per_resource_block * parallel_blocks / duration)
    results['max_transmission_rate'] = max_transmission_rate

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
    K_dB = -228.6

    power_received_dB = (
        params['link_margin_dB'] +
        to_dB(params['noise_temp_kelvin']) +
        K_dB +
        params['noise_figure_dB'] +
        to_dB(params['data_rate']) +
        params['SNR_per_bit_dB']
    )
    results['power_received_dB'] = power_received_dB

    power_transmitted_dB = (
        power_received_dB +
        params['path_loss_dB'] +
        params['feed_line_loss_dB'] +
        params['other_losses_dB'] +
        params['fade_margin_dB'] -
        params['transmitter_antenna_gain_dB'] -
        params['receiver_antenna_gain_dB'] -
        params['receiver_amp_gain_dB'] -
        params['transmitter_amp_gain_dB']
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
    from math import ceil, sqrt
    results = {}

    cluster_sizes = [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 28]
    NB = 6

    def _dB_to_linear(db):
        return 10 ** (db / 10)

    traffic_per_user = (params['avg_call_duration_min'] / 60) * params['avg_call_rate_per_user']
    total_traffic = traffic_per_user * params['max_users']
    results['traffic_per_user'] = round(traffic_per_user, 4)
    results['total_traffic'] = round(total_traffic, 2)

    max_distance = ((params['receiver_sensitivity'] / _dB_to_linear(params['P0'])) ** (-1 / params['path_loss_exponent'])) * params['d0']
    results['max_distance_reliable'] = round(max_distance, 2)

    max_cell_size = (3 * sqrt(3) / 2) * (max_distance ** 2)
    results['max_cell_size'] = round(max_cell_size, 2)

    total_cells = ceil(params['total_area'] / max_cell_size)
    results['total_number_of_cells'] = total_cells

    traffic_per_cell = total_traffic / total_cells
    results['traffic_per_cell'] = round(traffic_per_cell, 2)

    x = ((_dB_to_linear(params['sir']) * NB) ** (2 / params['path_loss_exponent'])) / 3
    cluster_size = next(N for N in cluster_sizes if N >= x)
    results['cluster_size'] = cluster_size

    # Use Erlang-B table (embedded dictionary)
    column_name = str(int(params['grade_of_service'] * 100)) + '%'
    if column_name not in ERLANG_B_TABLE:
        column_name = '1%'  # default fallback for now

    try:
        traffic_column = ERLANG_B_TABLE[column_name]
        for i, value in enumerate(traffic_column):
            if value >= traffic_per_cell:
                channels_required = ERLANG_B_TABLE['(N)'][i]
                break
        else:
            raise ValueError("Traffic value exceeds Erlang-B table range.")
    except Exception as e:
        raise ValueError("Erlang-B lookup failed: " + str(e))

    results['channels_required'] = int(channels_required)

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
        return f"\u274c Gemini Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
