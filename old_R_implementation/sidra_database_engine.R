# =======================================================================
# THE DEFINITIVE SIDRA METADATA ENGINE (V4 - Advanced Relational Query)
# =======================================================================
# Version: 4.0
#
# Purpose:
#   A comprehensive tool to build a complete local database of all SIDRA
#   table metadata and query it with a powerful, relational search function.
#
# Changes from V3.1:
#   - The main function is now split into two distinct, clear functions:
#     `build_sidra_database()` and `query_sidra_database()`.
#   - The `query_sidra_database()` function is completely overhauled. It now
#     takes a list of targeted queries, allowing for precise, multi-faceted
#     searches across the entire relational database.
# =======================================================================


# --- 0. SCRIPT SETUP: LIBRARIES AND CONFIGURATIONS ---
suppressPackageStartupMessages({
  library(rvest); library(xml2); library(dplyr); library(stringr);
  library(tidyr); library(purrr); library(future); library(furrr); library(progressr)
})

# --- GLOBAL CONFIGURATIONS ---
DB_FILE_PATH <- "sidra_metadata_database.rds"
TABLE_ID_RANGE <- 1:10000


# --- 1. BUILD MODE: The Engine to Create the Database (Unchanged but presented as a standalone function) ---

#' Builds a comprehensive local database of all SIDRA table metadata.
#'
#' This is a slow, one-time operation that performs parallel web scraping
#' and parsing for a given range of table IDs.
#'
#' @param db_path The file path where the final database will be saved.
#' @param id_range A numeric vector of table IDs to parse.
#' @return The created database object, invisibly.
build_sidra_database <- function(db_path = DB_FILE_PATH, id_range = TABLE_ID_RANGE) {
  
  # --- Internal Helper Function for Parsing ---
  .geo_code_mapping <- tibble(
    cod = c("n1", "n2", "n3", "n8", "n9", "n7", "n13", "n14", "n15", "n23", "n6", "n10", "n11", "n102"),
    cod2 = c("Brazil", "Region", "State", "MesoRegion", "MicroRegion", "MetroRegion", "MetroRegionDiv", "IRD", "UrbAglo", "PopArrang", "City", "District", "subdistrict", "Neighborhood")
  )
  
  .parse_single_table_metadata <- function(table_id, geo_mapping) {
    local_db <- list(Tables = tibble(table_id = integer(), table_desc = character(), period_raw = character()), Variables = tibble(variable_id = integer(), table_id = integer(), variable_desc_clean = character()), Geographies = tibble(table_id = integer(), geo_level_api_code = character(), geo_level_desc = character()), Classifications = tibble(classification_id_placeholder = integer(), table_id = integer(), classification_api_code = character(), classification_desc = character()), Categories = tibble(category_id = integer(), classification_id_placeholder = integer(), table_id = integer(), category_desc = character()))
    page <- tryCatch(xml2::read_html(paste0("http://api.sidra.ibge.gov.br/desctabapi.aspx?c=", table_id)), error = function(e) NULL)
    if (is.null(page)) { local_db$Tables <- tibble(table_id = table_id, table_desc = "ERROR: INVALID_OR_FAILED_ID", period_raw = ""); return(local_db) }
    local_db$Tables <- tibble(table_id = table_id, table_desc = page %>% html_node("#lblNomeTabela") %>% html_text(), period_raw = page %>% html_node("#lblPeriodoDisponibilidade") %>% html_text())
    all_html_tables <- page %>% html_nodes("table")
    if (length(all_html_tables) >= 2) {
      var_table_raw <- all_html_tables[[2]] %>% html_table(fill = TRUE)
      if(nrow(var_table_raw) > 0) { local_db$Variables <- var_table_raw %>% transmute(variable_id = as.integer(str_extract(.[[1]], "^[0-9]+")), table_id = table_id, variable_desc_clean = str_trim(str_replace_all(.[[1]], c("^[0-9]+" = "", " - casas decimais:  padrão = , máximo =" = "")))) }
    }
    n_codes <- all_html_tables %>% html_table(fill = TRUE, trim = TRUE) %>% unlist() %>% str_extract("N[0-9]+") %>% str_subset("N[0-9]+") %>% tolower()
    if (length(n_codes) > 0) {
      n_descs1 <- page %>% html_nodes("p+ #tabPrincipal span:nth-child(4)") %>% html_text(); n_descs2 <- page %>% html_nodes("p+ #tabPrincipal span:nth-child(5)") %>% html_text()
      if(length(n_descs1) == length(n_codes) && length(n_descs2) == length(n_codes)){ local_db$Geographies <- tibble(cod = n_codes, desc = paste(n_descs1, n_descs2)) %>% left_join(geo_mapping, by = "cod") %>% filter(!is.na(cod2)) %>% transmute(table_id = table_id, geo_level_api_code = cod2, geo_level_desc = desc) }
    }
    class_codes_raw <- all_html_tables %>% html_table(fill = TRUE, trim = TRUE) %>% unlist() %>% str_extract("\\C[0-9]+") %>% str_subset("\\C[0-9]+") %>% tolower()
    if (length(class_codes_raw) > 0) {
      class_descs1 <- page %>% html_nodes(".tituloLinha:nth-child(4)") %>% html_text(); class_descs2 <- page %>% html_nodes(".tituloLinha:nth-child(5)") %>% html_text()
      class_list <- list(); cat_list <- list()
      for (i in 0:(length(class_codes_raw) - 1)) {
        class_id_placeholder <- i + 1; class_api_code <- class_codes_raw[i + 1]; class_desc <- paste(class_descs1[i + 1], class_descs2[i + 1])
        class_list[[class_id_placeholder]] <- tibble(classification_id_placeholder = class_id_placeholder, table_id = table_id, classification_api_code = class_api_code, classification_desc = class_desc)
        cat_texts <- page %>% html_nodes(paste0("#lstClassificacoes_lblQuantidadeCategorias_", i, "+ ", "#tabPrincipal span")) %>% html_text()
        if (length(cat_texts) > 0) {
          cat_data_raw <- tibble(text = cat_texts) %>% mutate(is_code = str_detect(text, "^[0-9]+$"), group_id = cumsum(is_code))
          cat_list[[class_id_placeholder]] <- cat_data_raw %>% filter(group_id > 0) %>% group_by(group_id) %>% summarise(category_id = as.integer(first(text)), category_desc = paste(text[!is_code], collapse = " "), .groups = "drop") %>% select(-group_id) %>% mutate(classification_id_placeholder = class_id_placeholder, table_id = table_id)
        }
      }
      local_db$Classifications <- bind_rows(class_list); local_db$Categories <- bind_rows(cat_list)
    }
    return(local_db)
  }
  
  message("--- Starting BUILD process ---")
  message(paste("This will parse all tables in range", min(id_range), "to", max(id_range), "and save the result."))
  plan(multisession, workers = availableCores() - 1)
  message(paste("Using", future::nbrOfWorkers(), "parallel workers..."))
  with_progress({
    list_of_results <- furrr::future_map(
      .x = id_range,
      .f = ~ .parse_single_table_metadata(.x, geo_mapping = .geo_code_mapping),
      .options = furrr_options(seed = TRUE),
      .progress = TRUE
    )
  })
  plan(sequential)
  message("\nAssembling and normalizing the database...")
  db_raw <- list_of_results %>% transpose() %>% map(bind_rows)
  db <- list()
  db$Tables <- db_raw$Tables %>% distinct(); db$Variables <- db_raw$Variables %>% distinct(); db$Geographies <- db_raw$Geographies %>% distinct(); db$Classifications <- db_raw$Classifications %>% distinct(); db$Categories <- db_raw$Categories %>% distinct()
  message(paste("Saving complete database to", db_path))
  saveRDS(db, file = db_path)
  message("--- BUILD COMPLETE ---")
  return(invisible(db))
}


# --- 2. QUERY MODE: The Advanced Search Function ---

#' Queries the local SIDRA metadata database with advanced criteria.
#'
#' @param query_list A named list where names are one of "Tables", "Variables",
#'   "Classifications", "Categories", "Geographies". The values are regex
#'   patterns to search for in the description fields of those components.
#' @param filter_criteria A named list with optional keys: `min_munis`,
#'   `min_years`, `year_range` for numeric/date filtering.
#' @param db_path The file path to the pre-built database.
#' @return A tibble of matching tables.
query_sidra_database <- function(query_list = NULL,
                                 filter_criteria = NULL,
                                 db_path = DB_FILE_PATH) {
  
  message("--- Entering Advanced QUERY mode ---")
  if (!file.exists(db_path)) { stop(paste("Database file not found at '", db_path, "'. Please run build_sidra_database() first.", sep="")) }
  
  message("Loading database...")
  db <- readRDS(db_path)
  
  # Start with all valid table IDs
  candidate_ids <- db$Tables %>% filter(table_desc != "ERROR: INVALID_OR_FAILED_ID") %>% pull(table_id)
  
  message("Applying filters...")
  
  # --- Apply Structural Filters ---
  if (!is.null(filter_criteria$min_munis)) {
    geo_filtered_ids <- db$Geographies %>%
      filter(geo_level_api_code == "City") %>%
      mutate(num_munis = readr::parse_number(geo_level_desc, locale = readr::locale(grouping_mark = "."))) %>%
      filter(num_munis >= filter_criteria$min_munis) %>% pull(table_id) %>% unique()
    candidate_ids <- intersect(candidate_ids, geo_filtered_ids)
  }
  
  if (!is.null(filter_criteria$min_years) && !is.null(filter_criteria$year_range)) {
    time_filtered_ids <- db$Tables %>%
      filter(table_id %in% candidate_ids) %>%
      mutate(
        years = map(str_extract_all(period_raw, "[0-9]{4}"), as.integer),
        years_in_range = map_int(years, ~ sum(.x >= filter_criteria$year_range[1] & .x <= filter_criteria$year_range[2], na.rm = TRUE))
      ) %>%
      filter(years_in_range >= filter_criteria$min_years) %>% pull(table_id) %>% unique()
    candidate_ids <- intersect(candidate_ids, time_filtered_ids)
  }
  
  # --- Apply Relational Keyword Queries ---
  if (!is.null(query_list)) {
    list_of_matched_ids <- list()
    
    # This loop dynamically builds a list of table IDs that match each query
    for (scope in names(query_list)) {
      search_term <- query_list[[scope]]
      
      id_vector <- switch(
        scope,
        Tables = db$Tables %>% filter(str_detect(str_to_lower(table_desc), str_to_lower(search_term))) %>% pull(table_id),
        Variables = db$Variables %>% filter(str_detect(str_to_lower(variable_desc_clean), str_to_lower(search_term))) %>% pull(table_id),
        Classifications = db$Classifications %>% filter(str_detect(str_to_lower(classification_desc), str_to_lower(search_term))) %>% pull(table_id),
        Categories = db$Categories %>% filter(str_detect(str_to_lower(category_desc), str_to_lower(search_term))) %>% pull(table_id),
        Geographies = db$Geographies %>% filter(str_detect(str_to_lower(geo_level_desc), str_to_lower(search_term))) %>% pull(table_id),
        stop(paste("Invalid search scope:", scope))
      )
      list_of_matched_ids[[scope]] <- unique(id_vector)
    }
    
    # Find the intersection of all query results
    final_query_ids <- Reduce(intersect, list_of_matched_ids)
    candidate_ids <- intersect(candidate_ids, final_query_ids)
  }
  
  results <- db$Tables %>% filter(table_id %in% candidate_ids)
  
  if (nrow(results) == 0) {
    message("No tables found matching all specified criteria.")
    return(tibble())
  }
  
  message(paste("\n>>> Found", nrow(results), "matching tables. <<<"))
  return(results)
}


# --- 3. EXAMPLE USAGE ---

# **BUILD MODE: Run this ONCE to create your local database.**
# build_sidra_database()

# **QUERY MODE: Powerful and flexible querying.**

# Example 1: Find tables about households. This query is now precise and correct.
household_tables <- query_sidra_database(
  query_list = list(
    Variables = "Domicílios"
  ),
  filter_criteria = list(
    min_munis = 5000,
    min_years = 3,
    year_range = c(2000, 2024)
  )
)
print(household_tables)


# Example 2: Find tables about mortality that are broken down by BOTH sex and age.
# This demonstrates the power of the relational query.
mortality_tables <- query_sidra_database(
  query_list = list(
    Tables = "Óbitos",
    Classifications = "sexo|idade" # Search for classifications containing 'sexo' OR 'idade'
  )
)
print(mortality_tables)