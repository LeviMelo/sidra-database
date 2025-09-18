# =========================================================================
# SCRIPT: Metadata Export for All Integration Tables
# VERSION: 25.0
#
# PURPOSE:
#   This script addresses the critical failure of the previous version
#   by performing the essential first step: exporting the complete
#   `info_sidra()` metadata for every table involved in the integration
#   project. This provides the ground-truth needed for a correct
#   implementation.
#
# OPERATION:
#   1. Defines all unique table IDs.
#   2. Loops through each ID.
#   3. Captures the full output of `info_sidra()`.
#   4. Writes the captured output to a single file: "sidra_metadata_full.txt".
# =========================================================================

# --- 0. Setup ---
suppressPackageStartupMessages({
  library(sidrar)
  library(purrr)
})

# --- 1. Define All Unique Table IDs from the Full Scope ---
ALL_TABLE_IDS <- c(
  # Module 1: Household Anchors
  "2009", # Censo 2000, 2010
  "4712", # Censo 2022
  
  # # Module 2: Infrastructure
  # "1453", # Censo 2000
  # "3218", # Censo 2010
  # "6805", # Censo 2022
  
  # Module 3: Education
  "1383", # Censo 2010 (Literacy)
  "1972", # Censo 2000, 2010 (Attendance)
  
  # Module 4: Social (Labor)
  "2098", # Censo 2000, 2010
  
  # Module 5: Enhanced Vital Statistics
  "2609", # Births (2003-2022)
  "2654", # Deaths (2003-2004)
  "4412", # Marriages (2013-2014)
  "1695", # Divorces (2009-2010)
  
  # Additional Tables
  "6579", # EstimaPop Est. Pop. Res.
  "5938", # PIBmuni2010 GDP at Current Prices
  "1685", # CEMPRE Local Units & Enterprises (06-21)
  "9509", # CEMPRE Local Units & Enterprises (22)
  "2093", # CENSO 00 Pop. by Race/Age/Gender
  "9606"  # CENSO 10/22 Pop. by Race/Age/Gender
)

# --- 2. Define Output File ---
OUTPUT_FILENAME <- "sidra_metadata_full.txt"

# --- 3. Execute Metadata Export ---
message(paste("--- Starting metadata export for", length(ALL_TABLE_IDS), "tables... ---"))
message(paste("--- Output will be saved to:", OUTPUT_FILENAME, "---"))

# Open the file connection. 'w' will create or overwrite the file.
file_conn <- file(OUTPUT_FILENAME, "w")

# Use walk to loop through each ID and write to the file
walk(ALL_TABLE_IDS, function(table_id) {
  
  cat(paste("\n\n=========================================================\n"))
  cat(paste("### METADATA FOR TABLE:", table_id, "###\n"))
  cat(paste("=========================================================\n\n"))
  
  # Write header to the file
  cat(paste("\n\n=========================================================\n"), file = file_conn, append = TRUE)
  cat(paste("### METADATA FOR TABLE:", table_id, "###\n"), file = file_conn, append = TRUE)
  cat(paste("=========================================================\n\n"), file = file_conn, append = TRUE)
  
  # Get metadata and capture the printed output
  metadata_output <- tryCatch({
    capture.output(info_sidra(table_id))
  }, error = function(e) {
    paste("ERROR: Failed to fetch metadata for table", table_id, "-", e$message)
  })
  
  # Write the captured output to the file
  cat(metadata_output, file = file_conn, sep = "\n", append = TRUE)
  
})

# Close the file connection
close(file_conn)

message(paste("--- Metadata export complete. Please check the file:", OUTPUT_FILENAME, "---"))