"""
================================================================================
VAST.AI DATA COLLECTION SCRIPT
================================================================================

This script collects GPU rental data from Vast.ai API for training ML models.

Usage:
    1. Set your API key: export VAST_API_KEY="your-api-key"
    2. Run: python collect_data.py

Data collected:
    - GPU specs (name, count, RAM, FLOPS)
    - Pricing (hourly rate, storage cost)
    - Host info (reliability, location, datacenter)
    - Network (upload/download speed)
    - Status (rented or available)

Author: Wenjia
GitHub: https://github.com/Wenjia000/Muti-Agent_AI_System
================================================================================
"""

import requests
import json
import os
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Configuration
VAST_API_URL = "https://console.vast.ai/api/v0/bundles/"

# Data storage path
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "vast_historical_data.json"

# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

def get_api_key():
    """Get API key from environment variable."""
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        raise ValueError(
            "VAST_API_KEY not found!\n"
            "Set it with: export VAST_API_KEY='your-api-key'\n"
            "Get your key from: https://cloud.vast.ai/account/"
        )
    return api_key


def fetch_vast_offers(api_key: str) -> list:
    """
    Fetch all available GPU offers from Vast.ai API.
    
    Returns:
        list: List of offer dictionaries
    """
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Search for all verified, rentable offers
    payload = {
        "limit": 2000,
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "order": [["dph_total", "asc"]]
    }
    
    print("📡 Fetching data from Vast.ai API...")
    response = requests.post(VAST_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    data = response.json()
    offers = data.get("offers", [])
    print(f"✅ Fetched {len(offers)} offers")
    
    return offers


def process_offer(offer: dict) -> dict:
    """
    Extract relevant fields from a single offer.
    
    Args:
        offer: Raw offer data from API
        
    Returns:
        dict: Processed offer with selected fields
    """
    now = datetime.now()
    
    return {
        # Timestamp
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "hour": now.hour,
        "day_of_week": now.weekday(),  # 0=Monday, 6=Sunday
        "is_weekend": now.weekday() >= 5,
        
        # GPU Information
        "gpu_name": offer.get("gpu_name"),
        "num_gpus": offer.get("num_gpus"),
        "gpu_ram": offer.get("gpu_ram"),  # GB
        "gpu_total_ram": offer.get("gpu_totalram"),  # Total GPU RAM
        "total_flops": offer.get("total_flops"),
        "dlperf": offer.get("dlperf"),  # Deep learning performance
        "dlperf_per_dphtotal": offer.get("dlperf_per_dphtotal"),  # Performance per dollar
        
        # Host Information
        "machine_id": offer.get("machine_id"),
        "host_id": offer.get("host_id"),
        "reliability": offer.get("reliability2", offer.get("reliability")),
        "geolocation": offer.get("geolocation"),
        "datacenter": offer.get("datacenter", False),
        "verified": offer.get("verified", False),
        
        # Network
        "inet_down": offer.get("inet_down"),  # Download speed Mbps
        "inet_up": offer.get("inet_up"),  # Upload speed Mbps
        "static_ip": offer.get("static_ip", False),
        "direct_port_count": offer.get("direct_port_count"),
        
        # CPU & Storage
        "cpu_cores": offer.get("cpu_cores"),
        "cpu_ghz": offer.get("cpu_ghz"),
        "cpu_name": offer.get("cpu_name"),
        "disk_space": offer.get("disk_space"),  # GB
        "disk_bw": offer.get("disk_bw"),  # Disk bandwidth
        
        # Pricing (IMPORTANT!)
        "dph_total": offer.get("dph_total"),  # Price per hour ($/hr)
        "dph_base": offer.get("dph_base"),  # Base GPU price
        "storage_cost": offer.get("storage_cost"),  # Storage $/GB/month
        "inet_up_cost": offer.get("inet_up_cost"),  # Upload cost
        "inet_down_cost": offer.get("inet_down_cost"),  # Download cost
        
        # Rental Status (THIS IS THE TRAINING LABEL!)
        "rented": offer.get("rented", False),
        "num_renting": offer.get("num_renting", 0),
        
        # Duration
        "duration": offer.get("duration"),  # Max rental duration in days
        "min_bid": offer.get("min_bid"),
        
        # Other
        "cuda_max_good": offer.get("cuda_max_good"),
        "compute_cap": offer.get("compute_cap"),  # CUDA compute capability
        "pci_gen": offer.get("pci_gen"),
        "pcie_bw": offer.get("pcie_bw"),
    }


def collect_data() -> list:
    """
    Main function to collect and process all Vast.ai data.
    
    Returns:
        list: List of processed offer records
    """
    api_key = get_api_key()
    offers = fetch_vast_offers(api_key)
    
    print("📊 Processing offers...")
    records = []
    for offer in offers:
        try:
            record = process_offer(offer)
            records.append(record)
        except Exception as e:
            print(f"⚠️ Error processing offer: {e}")
            continue
    
    print(f"✅ Processed {len(records)} records")
    return records


def load_existing_data() -> list:
    """Load existing data from file if it exists."""
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []


def save_data(records: list):
    """Save records to JSON file."""
    DATA_DIR.mkdir(exist_ok=True)
    
    with open(DATA_FILE, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"💾 Saved {len(records)} records to {DATA_FILE}")


def get_summary_stats(records: list) -> dict:
    """Calculate summary statistics for the collected data."""
    if not records:
        return {}
    
    # Get latest batch (same timestamp)
    latest_timestamp = records[-1]["timestamp"]
    latest_batch = [r for r in records if r["timestamp"] == latest_timestamp]
    
    # Count by GPU type
    gpu_counts = {}
    rented_counts = {}
    price_by_gpu = {}
    
    for r in latest_batch:
        gpu = r.get("gpu_name", "Unknown")
        gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
        
        if r.get("rented"):
            rented_counts[gpu] = rented_counts.get(gpu, 0) + 1
        
        if gpu not in price_by_gpu:
            price_by_gpu[gpu] = []
        if r.get("dph_total"):
            price_by_gpu[gpu].append(r["dph_total"])
    
    # Calculate rental rates and median prices
    stats = {}
    for gpu in gpu_counts:
        rented = rented_counts.get(gpu, 0)
        total = gpu_counts[gpu]
        prices = sorted(price_by_gpu.get(gpu, []))
        
        stats[gpu] = {
            "total": total,
            "rented": rented,
            "available": total - rented,
            "rental_rate": round(rented / total * 100, 1) if total > 0 else 0,
            "price_min": round(min(prices), 3) if prices else None,
            "price_median": round(prices[len(prices)//2], 3) if prices else None,
            "price_max": round(max(prices), 3) if prices else None,
        }
    
    return stats


def print_summary(stats: dict):
    """Print summary of collected data."""
    print("\n" + "=" * 70)
    print("📊 DATA COLLECTION SUMMARY")
    print("=" * 70)
    print(f"{'GPU':<20} {'Total':<8} {'Rented':<8} {'Rate':<8} {'Price Range':<20}")
    print("-" * 70)
    
    # Sort by total count
    sorted_gpus = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)[:15]
    
    for gpu, data in sorted_gpus:
        price_range = f"${data['price_min']:.2f} - ${data['price_max']:.2f}" if data['price_min'] else "N/A"
        print(f"{gpu:<20} {data['total']:<8} {data['rented']:<8} {data['rental_rate']:<7}% {price_range:<20}")
    
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("🚀 VAST.AI DATA COLLECTION")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Collect new data
        new_records = collect_data()
        
        if not new_records:
            print("❌ No data collected")
            return
        
        # Load existing data and append
        existing_data = load_existing_data()
        print(f"📂 Existing records: {len(existing_data)}")
        
        # Append new records
        all_records = existing_data + new_records
        
        # Save all data
        save_data(all_records)
        
        # Print summary
        stats = get_summary_stats(all_records)
        print_summary(stats)
        
        print(f"\n✅ Collection complete!")
        print(f"📊 Total records in database: {len(all_records)}")
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
