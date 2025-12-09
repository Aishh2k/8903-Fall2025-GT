import sys
import os
import csv
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

print(f"Processing RFCs from newest to oldest to get 150 unique affiliations...")

# Extract affiliations until we get 150 unique ones
unique_affiliations = set()
output_rows = []
rfcs_processed = 0

for doc in all_rfcs:
    try:
        rfc_number = doc.name if hasattr(doc, 'name') else 'Unknown'
        authors = dt.document_authors(doc)
        
        if authors:
            for a in authors:
                if hasattr(a, 'affiliation') and a.affiliation:
                    aff = a.affiliation
                    if isinstance(aff, str) and aff.strip():
                        clean_aff = aff.strip()
                        
                        # Only add if this affiliation is new
                        if clean_aff not in unique_affiliations:
                            unique_affiliations.add(clean_aff)
                            output_rows.append([rfc_number, clean_aff])
                            
                            # Stop when we reach 150 unique affiliations
                            if len(unique_affiliations) >= 150:
                                break
            
            if len(unique_affiliations) >= 150:
                break
        
        rfcs_processed += 1

    except Exception as e:
        continue

# Save CSV file
output_file = os.path.join(curr_dir, "affiliations_raw.csv")

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["rfc_id", "original_affiliation"])
    writer.writerows(output_rows)

print(f"Processed {rfcs_processed} RFCs to find 150 unique affiliations")
print(f"Total unique affiliations extracted: {len(output_rows)}")
print(f"Saved to: {output_file}")