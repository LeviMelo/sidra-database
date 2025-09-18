# =======================================================================
# OPTIMIZED SIDRA TABLE SEARCH SCRIPT
# =======================================================================
# Purpose:
#   To efficiently find all SIDRA tables that meet geographic and
#   temporal criteria by using parallel processing. This version is
#   significantly faster than the previous brute-force attempt.
#
# Changes:
#   - Removed the unnecessary Sys.sleep() delay.
#   - Implemented parallel processing using the 'furrr' and 'future'
#     packages to check multiple tables simultaneously.
# =======================================================================

# --- Step 0: Setup ---
message("--- Step 0: Loading Libraries & Defining Criteria ---")
suppressPackageStartupMessages({
  library(sidrar)
  library(dplyr)
  library(stringr)
  library(readr)
  library(furrr) # For parallel mapping
  library(future) # For setting up the parallel plan
})

# Define the criteria
MIN_MUNICIPIOS <- 5000
MIN_YEARS_IN_RANGE <- 2
START_YEAR <- 2000
END_YEAR <- 2022

# --- Step 1: The Validated Parser Function (No changes needed) ---
message("--- Step 1: Using the validated parser function ---")

check_sidra_table_criteria <- function(table_id) {
  tryCatch({
    metadata <- info_sidra(table_id)
    
    # Geographic Validation
    geo_df <- metadata$geo
    meets_geo <- FALSE
    num_munis <- 0
    if (!is.null(geo_df) && nrow(geo_df) > 0) {
      muni_row <- geo_df[str_detect(geo_df$desc, "^Município"), ]
      if (nrow(muni_row) == 1) {
        num_munis <- readr::parse_number(
          muni_row$desc,
          locale = readr::locale(grouping_mark = ".")
        )
        if (!is.na(num_munis) && num_munis >= MIN_MUNICIPIOS) {
          meets_geo <- TRUE
        }
      }
    }
    
    # Temporal Validation
    period_str <- metadata$period
    meets_time <- FALSE
    years_in_range_str <- ""
    if (!is.null(period_str)) {
      available_years <- as.integer(unlist(strsplit(period_str, ", ")))
      years_in_range <- available_years[available_years >= START_YEAR & available_years <= END_YEAR]
      if (length(years_in_range) >= MIN_YEARS_IN_RANGE) {
        meets_time <- TRUE
        years_in_range_str <- paste(years_in_range, collapse = ", ")
      }
    }
    
    tibble(
      table_id = table_id,
      meets_criteria = meets_geo & meets_time,
      description = metadata$table,
      num_munis = num_munis,
      years_in_range = years_in_range_str
    )
  }, error = function(e) {
    # Return a failure record for this ID, but don't stop the process
    tibble(
      table_id = table_id,
      meets_criteria = FALSE,
      description = "ERROR: Failed to fetch or parse.",
      num_munis = 0,
      years_in_range = ""
    )
  })
}

# --- Step 2: High-Speed Parallel Search ---
message("\n--- Step 2: Starting High-Speed Parallel Search ---")

# Define the range of table IDs to check
TABLE_ID_RANGE <- 1:10000

# Set up the parallel processing plan.
# This will use all available CPU cores on your machine minus one.
# You can change the number of workers if you wish, e.g., plan(multisession, workers = 4)
plan(multisession, workers = availableCores() - 1)
message(paste("Using", future::nbrOfWorkers(), "parallel workers to accelerate the search."))

# Use future_map_dfr for parallel execution.
# It automatically applies the function to the list of IDs across workers.
# The .progress = TRUE argument provides a real-time progress bar.
all_results <- furrr::future_map_dfr(
  .x = TABLE_ID_RANGE,
  .f = ~ check_sidra_table_criteria(.x),
  .options = furrr_options(seed = TRUE), # for reproducibility
  .progress = TRUE # This provides the progress bar
)

# It's good practice to return to a sequential plan after the parallel task is done
plan(sequential)


# --- Step 3: Report Final Results ---
message("\n\n--- Step 3: Analysis Complete. Reporting Results ---")

final_matches <- all_results %>%
  filter(meets_criteria == TRUE) %>%
  arrange(table_id)

message(paste("\n>>> Found", nrow(final_matches), "tables that meet all criteria <<<"))

if (nrow(final_matches) > 0) {
  message("The following tables have data for 5000+ municipalities and span at least 2 years between 2000-2022:")
  # Use print.data.frame to ensure the full description is shown
  print(as.data.frame(final_matches))
} else {
  message("No tables were found that match all the specified criteria.")
}