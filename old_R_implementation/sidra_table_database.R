# =======================================================================
# OPTIMIZED SIDRA Metadata Database Builder (V2 - Robust Parser)
# =======================================================================
# Purpose:
#   To robustly parse SIDRA metadata in parallel, handling irregular
#   HTML structures that caused the previous version to crash.
#
# Change:
#   - Replaced the brittle category parser with an intelligent one that
#     identifies codes and descriptions by content, not position.
# =======================================================================

# --- Step 0: Setup ---
message("--- Step 0: Loading Libraries ---")
suppressPackageStartupMessages({
  library(rvest)
  library(xml2)
  library(dplyr)
  library(stringr)
  library(tidyr)
  library(purrr)
  library(future)
  library(furrr)
})

# --- Step 1: Verification and Helper Definition ---
message("--- Step 1: Defining the ROBUST Parser Function ---")

if (!exists("final_matches") || !is.data.frame(final_matches) || nrow(final_matches) == 0) {
  stop("The 'final_matches' object is empty or does not exist.")
}

geo_code_mapping <- tibble(
  cod = c("n1", "n2", "n3", "n8", "n9", "n7", "n13", "n14", "n15", "n23", "n6", "n10", "n11", "n102"),
  cod2 = c("Brazil", "Region", "State", "MesoRegion", "MicroRegion", "MetroRegion", "MetroRegionDiv", "IRD", "UrbAglo", "PopArrang", "City", "District", "subdistrict", "Neighborhood")
)

parse_single_table_metadata <- function(table_id) {
  
  local_database <- list(
    Tables = tibble(table_id = integer(), table_desc = character(), period_raw = character()),
    Variables = tibble(variable_id = integer(), table_id = integer(), variable_desc_raw = character(), variable_desc_clean = character(), variable_unit = character()),
    Geographies = tibble(geo_id_placeholder = integer(), table_id = integer(), geo_level_api_code = character(), geo_level_desc = character()),
    Classifications = tibble(classification_id_placeholder = integer(), table_id = integer(), classification_api_code = character(), classification_desc = character()),
    Categories = tibble(category_id = integer(), classification_id_placeholder = integer(), table_id = integer(), category_desc = character(), category_unit = character())
  )
  
  page_content <- tryCatch(
    xml2::read_html(paste0("http://api.sidra.ibge.gov.br/desctabapi.aspx?c=", table_id)),
    error = function(e) NULL
  )
  
  if (is.null(page_content)) {
    local_database$Tables <- tibble(table_id = table_id, table_desc = "ERROR: Table ID does not exist or failed to load.", period_raw = "")
    return(local_database)
  }
  
  # Parse main table info
  local_database$Tables <- tibble(
    table_id = table_id,
    table_desc = page_content %>% html_node("#lblNomeTabela") %>% html_text(),
    period_raw = page_content %>% html_node("#lblPeriodoDisponibilidade") %>% html_text()
  )
  
  # Parse Variables
  var_table_raw <- page_content %>% html_nodes("table") %>% .[[2]] %>% html_table(fill = TRUE)
  if(nrow(var_table_raw) > 0) {
    local_database$Variables <- var_table_raw %>%
      transmute(
        variable_id = as.integer(str_extract(.[[1]], "^[0-9]+")),
        table_id = table_id,
        variable_desc_raw = str_trim(str_replace(.[[1]], "^[0-9]+", "")),
        variable_desc_clean = str_replace(variable_desc_raw, " - casas decimais:  padrão = , máximo =", ""),
        variable_unit = str_extract(variable_desc_raw, "(?<=\\().*?(?=\\))|(?<=\\[).*?(?=\\])") %>% str_trim()
      )
  }
  
  # Parse Geographies
  n_codes <- page_content %>% html_nodes("table") %>% html_table(fill = TRUE, trim = TRUE) %>% unlist() %>% str_extract("N[0-9]+") %>% str_subset("N[0-9]+") %>% tolower()
  if (length(n_codes) > 0) {
    n_descs_part1 <- page_content %>% html_nodes("p+ #tabPrincipal span:nth-child(4)") %>% html_text()
    n_descs_part2 <- page_content %>% html_nodes("p+ #tabPrincipal span:nth-child(5)") %>% html_text()
    if(length(n_descs_part1) == length(n_codes) && length(n_descs_part2) == length(n_codes)){
      n_geo_raw <- tibble(cod = n_codes, desc = paste(n_descs_part1, n_descs_part2))
      local_database$Geographies <- n_geo_raw %>% 
        left_join(geo_code_mapping, by = "cod") %>%
        filter(!is.na(cod2)) %>%
        transmute(geo_id_placeholder = row_number(), table_id = table_id, geo_level_api_code = cod2, geo_level_desc = desc)
    }
  }
  
  # Parse Classifications and Categories
  class_codes_raw <- page_content %>% html_nodes("table") %>% html_table(fill = TRUE, trim = TRUE) %>% unlist() %>% str_extract("\\C[0-9]+") %>% str_subset("\\C[0-9]+") %>% tolower()
  if (length(class_codes_raw) > 0) {
    class_descs_part1 <- page_content %>% html_nodes(".tituloLinha:nth-child(4)") %>% html_text()
    class_descs_part2 <- page_content %>% html_nodes(".tituloLinha:nth-child(5)") %>% html_text()
    
    class_list <- list()
    cat_list <- list()
    
    for (i in 0:(length(class_codes_raw) - 1)) {
      class_id_placeholder <- i + 1
      class_api_code <- class_codes_raw[i + 1]
      class_desc <- paste(class_descs_part1[i + 1], class_descs_part2[i + 1])
      class_list[[class_id_placeholder]] <- tibble(classification_id_placeholder = class_id_placeholder, table_id = table_id, classification_api_code = class_api_code, classification_desc = class_desc)
      
      cat_texts <- page_content %>% html_nodes(paste0("#lstClassificacoes_lblQuantidadeCategorias_", i, "+ ", "#tabPrincipal span")) %>% html_text()
      
      # ================== ROBUST PARSER LOGIC START ==================
      if (length(cat_texts) > 0) {
        cat_data_raw <- tibble(text = cat_texts) %>%
          mutate(is_code = str_detect(text, "^[0-9]+$")) %>%
          mutate(group_id = cumsum(is_code))
        
        cat_list[[class_id_placeholder]] <- cat_data_raw %>%
          filter(group_id > 0) %>% # Ignore any text before the first code
          group_by(group_id) %>%
          summarise(
            category_id = as.integer(first(text)),
            category_desc = paste(text[!is_code], collapse = " "),
            .groups = "drop"
          ) %>%
          select(-group_id) %>%
          mutate(
            classification_id_placeholder = class_id_placeholder,
            table_id = table_id,
            category_unit = if_else(str_detect(category_desc, "^(Quilogramas|Toneladas|Mil|Pessoas|Hectares|Cabeças|Dúzias)"), category_desc, NA_character_)
          )
      }
      # =================== ROBUST PARSER LOGIC END ===================
    }
    local_database$Classifications <- bind_rows(class_list)
    local_database$Categories <- bind_rows(cat_list)
  }
  
  return(local_database)
}

# --- Step 2: High-Speed Parallel Execution ---
message(paste("\n--- Step 2: Starting High-Speed Parallel Parsing for", nrow(final_matches), "tables ---"))
table_ids_to_process <- final_matches$table_id

plan(multisession, workers = availableCores() - 1)
message(paste("Using", future::nbrOfWorkers(), "parallel workers to accelerate the process."))

list_of_results <- furrr::future_map(
  .x = table_ids_to_process,
  .f = ~ parse_single_table_metadata(.x),
  .options = furrr_options(seed = TRUE),
  .progress = TRUE
)

plan(sequential)

# --- Step 3: Final Assembly ---
message("\n\n--- Step 3: Assembling the final database from parallel results ---")

sidra_database_raw <- list_of_results %>%
  purrr::transpose() %>%
  purrr::map(bind_rows)

# --- Step 4: Final Normalization (V2 - Corrected Joins) ---
message("--- Step 4: Normalizing the final database structure ---")

# Start with a clean list
sidra_database <- list()

# Normalize Tables - No change needed here
sidra_database$Tables <- sidra_database_raw$Tables %>%
  filter(str_detect(table_desc, "ERROR", negate = TRUE)) %>%
  distinct()

# Normalize Variables - No change needed here
sidra_database$Variables <- sidra_database_raw$Variables %>%
  select(-table_id) %>%
  distinct(variable_id, .keep_all = TRUE)

# Build junction table for Table-Variables - No change needed here
sidra_database$TableVariables <- sidra_database_raw$Variables %>%
  select(table_id, variable_id) %>%
  distinct()

# Normalize Classifications and their junction table
# This part is the same
sidra_database$Classifications <- sidra_database_raw$Classifications %>%
  group_by(classification_api_code, classification_desc) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(classification_id = row_number()) %>%
  select(classification_id, classification_api_code, classification_desc)

# ============================ THE FIX IS HERE ============================
# The logic for joining classifications and categories is now more robust.

# 1. Create a mapping table that links the placeholder to the final classification_id
placeholder_to_final_id_map <- sidra_database_raw$Classifications %>%
  left_join(sidra_database$Classifications, by = c("classification_api_code", "classification_desc")) %>%
  select(table_id, classification_id_placeholder, classification_id)

# 2. Build the final, clean TableClassifications junction table
sidra_database$TableClassifications <- placeholder_to_final_id_map %>%
  select(table_id, classification_id) %>%
  distinct()

# 3. Normalize Categories using the precise mapping table
# This join is now one-to-many and will not produce a warning.
sidra_database$Categories <- sidra_database_raw$Categories %>%
  left_join(placeholder_to_final_id_map, by = c("table_id", "classification_id_placeholder")) %>%
  filter(!is.na(classification_id)) %>%
  select(classification_id, category_id, category_desc, category_unit) %>%
  distinct()
# ============================ END OF FIX ============================

# Normalize Geographies
sidra_database$Geographies <- sidra_database_raw$Geographies %>%
  select(-geo_id_placeholder) %>%
  distinct()

# --- Step 5: Final Report (Unchanged) ---
message("\n--- DATABASE CREATION COMPLETE ---")
message("A list named `sidra_database` has been created with the following tables:")
# Reorder for clarity
table_order <- c("Tables", "Variables", "Classifications", "Categories", "Geographies", "TableVariables", "TableClassifications")
sidra_database <- sidra_database[table_order]

for (name in names(sidra_database)) {
  if (!is.null(sidra_database[[name]])) {
    message(paste0(" - $", name, ": ", nrow(sidra_database[[name]]), " rows"))
  }
}

message("\nTo inspect the first few rows of the 'Categories' table, for example, run:")
message("head(sidra_database$Categories)")