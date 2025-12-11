import sys
import os
import csv
import time
curr_dir = os.path.dirname(os.path.abspath(__file__))

from ietfdata.datatracker import DataTracker, DTBackendLive

dt = DataTracker(DTBackendLive())
rfc_doctype = dt.document_type_from_slug("rfc")

print("Fetching all RFC documents...")
all_rfcs = list(dt.documents(doctype=rfc_doctype))
print(f"Total RFCs found: {len(all_rfcs)}")

def get_rfc_number(doc):
    try:
        name = doc.name if hasattr(doc, 'name') else ''
        if name.startswith('rfc'):
            return int(name[3:])
        return 0
    except:
        return 0

all_rfcs.sort(key=get_rfc_number, reverse=True)

print(f"Processing RFCs to get 150 unique addresses...")

# Extract addresses until we get 150 unique ones
unique_addresses = set()
output_rows = []
rfcs_processed = 0

for doc in all_rfcs:
    try:
        rfc_number = doc.name if hasattr(doc, 'name') else 'Unknown'
        authors = dt.document_authors(doc)
        
        rfcs_processed += 1
        
        if rfcs_processed <= 3:
            if authors:
                for i, a in enumerate(authors):
                    print(f"  Author {i}: {dir(a)}")
                    if hasattr(a, 'address'):
                        print(f"    address: {a.address}")
                    if hasattr(a, 'country'):
                        print(f"    country: {a.country}")
                    if hasattr(a, 'person'):
                        print(f"    person: {a.person}")
                        if a.person and hasattr(a.person, 'address'):
                            print(f"      person.address: {a.person.address}")
            else:
                print(f"  No authors found")
        
        if authors:
            for a in authors:
                address = None
                
                if hasattr(a, 'address') and a.address:
                    address = a.address
                elif hasattr(a, 'country') and a.country:
                    address = a.country
                elif hasattr(a, 'person') and a.person:
                    if hasattr(a.person, 'address') and a.person.address:
                        address = a.person.address
                    elif hasattr(a.person, 'country') and a.person.country:
                        address = a.person.country
                
                if address and isinstance(address, str) and address.strip():
                    clean_addr = address.strip()
                    
                    # Only add if this address is new
                    if clean_addr not in unique_addresses:
                        unique_addresses.add(clean_addr)
                        output_rows.append([rfc_number, clean_addr])
                        print(f"Found {len(unique_addresses)}/150 unique addresses from {rfc_number}")
                        
                        if len(unique_addresses) >= 150:
                            break
                    
                    break
            
            if len(unique_addresses) >= 150:
                break

    except Exception as e:
        print(f"Error processing {doc.name}: {e}")
        continue

output_file = os.path.join(curr_dir, "address_raw.csv")

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["rfc_id", "original_address"])
    writer.writerows(output_rows)

print(f"\nProcessed {rfcs_processed} RFCs to find 150 unique addresses")
print(f"Saved to: {output_file}")