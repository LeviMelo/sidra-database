# =======================================================================
# THE UNIFIED SIDRA DATABASE AND SEARCH ENGINE (V8.0 - COMPONENT SEARCH)
# =======================================================================
# Version: 8.0
#
# Purpose:
#   The definitive script, redesigned to support targeted semantic search
#   on individual table components (e.g., Variables, Classifications).
#
# Key Improvements in V8.0:
#   - REDESIGNED SEMANTIC INDEX: The `build_semantic_database` function
#     now creates separate, dedicated embeddings for Table Titles,
#     Variables, and Classifications.
#   - POWERFUL TARGETED QUERIES: The `query_sidra` function now accepts a
#     `semantic_scope` parameter ("Tables", "Variables", "Classifications")
#     to perform surgical, context-aware searches.
#   - This directly enables the requested feature: finding tables that
#     contain variables semantically similar to a given concept.
# =======================================================================



# --- 0. SCRIPT SETUP: LIBRARIES AND GLOBAL CONFIGURATIONS ---

# CRITICAL: SET YOUR PYTHON PATH
Sys.setenv(RETICULATE_PYTHON = "C:/Users/Galaxy/miniconda3/envs/litscape/python.exe")

suppressPackageStartupMessages({
  library(rvest); library(xml2); library(dplyr); library(stringr);
  library(tidyr); library(purrr); library(future); library(furrr); library(progressr);
  library(reticulate);
})

PROJECT_DIR <- "C:/Users/Galaxy/LEVI/projects/R/demographic_basis_sidra_database"
RAW_DB_PATH <- file.path(PROJECT_DIR, "sidra_metadata_database.rds")
SEMANTIC_DB_PATH <- file.path(PROJECT_DIR, "sidra_semantic_database.rds")


# =======================================================================
# --- SECTION 1: RAW DATABASE ENGINE (ROBUST & TRANSPARENT) ---
# =======================================================================

#' Builds a comprehensive local database of all SIDRA table metadata.
build_sidra_database <- function(db_path = RAW_DB_PATH, id_range = 1:12000) {
  
  .geo_code_mapping <- tibble(cod = c("n1", "n2", "n3", "n8", "n9", "n7", "n13", "n14", "n15", "n23", "n6", "n10", "n11", "n102"), cod2 = c("Brazil", "Region", "State", "MesoRegion", "MicroRegion", "MetroRegion", "MetroRegionDiv", "IRD", "UrbAglo", "PopArrang", "City", "District", "subdistrict", "Neighborhood"))
  
  .parse_single_table_metadata <- function(table_id, geo_mapping) {
    Sys.sleep(runif(1, 0.05, 0.25))
    
    local_db <- list(Tables = tibble(table_id = integer(), table_desc = character(), period_raw = character()), Variables = tibble(variable_id = integer(), table_id = integer(), variable_desc_clean = character()), Geographies = tibble(table_id = integer(), geo_level_api_code = character(), geo_level_desc = character()), Classifications = tibble(classification_id_placeholder = integer(), table_id = integer(), classification_api_code = character(), classification_desc = character()), Categories = tibble(category_id = integer(), classification_id_placeholder = integer(), table_id = integer(), category_desc = character()))
    
    # --- DEFINITIVE FIX: Wrap the web request in a timeout ---
    page <- NULL
    tryCatch({
      page <- R.utils::withTimeout({
        xml2::read_html(paste0("http://api.sidra.ibge.gov.br/desctabapi.aspx?c=", table_id))
      }, timeout = 30, onTimeout = "error") # Give up after 60 seconds
    }, TimeoutException = function(ex) {
      # This block will run if a timeout occurs
      message(paste("\n[WARNING] Timeout occurred for table_id:", table_id, "- Skipping."))
      page <<- NULL # Ensure page is NULL
    }, error = function(e) {
      # This catches other errors, like standard HTTP errors
      page <<- NULL
    })
    
    if (is.null(page)) { local_db$Tables <- tibble(table_id = table_id, table_desc = "ERROR: FAILED_OR_TIMED_OUT", period_raw = ""); return(local_db) }
    
    # The rest of the parsing logic is unchanged
    local_db$Tables <- tibble(table_id = table_id, table_desc = page %>% html_node("#lblNomeTabela") %>% html_text(), period_raw = page %>% html_node("#lblPeriodoDisponibilidade") %>% html_text())
    # ... (rest of the parsing logic from V7.1)
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
  
  message("--- Starting BUILD process for Raw Database (V7.2 - Hardened with Timeouts) ---")
  plan(multisession, workers = availableCores() - 1)
  message(paste("Using", future::nbrOfWorkers(), "parallel workers..."))
  
  handlers(handler_progress(format = "[:bar] :percent, ETA: :eta"))
  
  with_progress({
    p <- progressor(steps = length(id_range))
    list_of_results <- furrr::future_map(id_range, ~{
      p()
      .parse_single_table_metadata(.x, geo_mapping = .geo_code_mapping)
    }, .options = furrr_options(seed = TRUE))
  })
  plan(sequential)
  
  message("\n[INFO] Parallel scraping complete. Assembling data...")
  message(paste("  -", length(list_of_results), "total items returned."))
  message("  - Transposing list of lists (this can be slow)...")
  transposed_list <- purrr::transpose(list_of_results)
  message("  - Transposing complete.")
  message("  - Binding rows for each data component...")
  db <- list()
  db$Tables <- bind_rows(transposed_list$Tables)
  message("    - 'Tables' bound.")
  db$Variables <- bind_rows(transposed_list$Variables)
  message("    - 'Variables' bound.")
  db$Geographies <- bind_rows(transposed_list$Geographies)
  message("    - 'Geographies' bound.")
  db$Classifications <- bind_rows(transposed_list$Classifications)
  message("    - 'Classifications' bound.")
  db$Categories <- bind_rows(transposed_list$Categories)
  message("    - 'Categories' bound.")
  message("  - Row binding complete.")
  
  message("Normalizing and saving the database...")
  db$Tables <- db$Tables %>% distinct()
  db$Variables <- db$Variables %>% distinct()
  
  saveRDS(db, file = db_path)
  message(paste("--- RAW DATABASE BUILD COMPLETE. Saved to:", db_path, "---"))
  return(invisible(db))
}

# =======================================================================
# --- SECTION 2: SEMANTIC INDEX ENGINE (UPGRADED TO V9.0) ---
# =======================================================================

#' Builds a multi-level semantic database from the raw metadata.
#'
#' This new version (V9.0) creates four separate, searchable semantic indexes for
#' Table Titles, Variables, Classifications, and now Categories.
#'
build_semantic_database <- function(raw_db_path = RAW_DB_PATH,
                                    semantic_db_path = SEMANTIC_DB_PATH, # Ensure this points to your new v9 path
                                    model_name = "paraphrase-multilingual-mpnet-base-v2") {
  
  message("--- Entering BUILD mode for Multi-Level Semantic Database (V9.0) ---")
  if (!file.exists(raw_db_path)) stop("Raw database not found.")
  db_raw <- readRDS(raw_db_path)
  
  message("Initializing Python environment...")
  SentenceTransformer <- reticulate::import("sentence_transformers")$SentenceTransformer
  model <- SentenceTransformer(model_name)
  
  embeddings_list <- list()
  
  # --- 1. Embed Table Titles ---
  corpus_tables <- db_raw$Tables %>% filter(table_desc != "ERROR: FAILED_OR_TIMED_OUT")
  if(nrow(corpus_tables) > 0) {
    message(paste("Generating embeddings for", nrow(corpus_tables), "Table Titles..."))
    embeddings_list$Tables <- model$encode(corpus_tables$table_desc, show_progress_bar = TRUE)
    rownames(embeddings_list$Tables) <- corpus_tables$table_id
  }
  
  # --- 2. Embed Variables ---
  corpus_variables <- db_raw$Variables %>% distinct(variable_id, .keep_all = TRUE)
  if(nrow(corpus_variables) > 0) {
    message(paste("Generating embeddings for", nrow(corpus_variables), "Variables..."))
    embeddings_list$Variables <- model$encode(corpus_variables$variable_desc_clean, show_progress_bar = TRUE)
    rownames(embeddings_list$Variables) <- corpus_variables$variable_id
  }
  
  # --- 3. Embed Classifications ---
  corpus_class <- db_raw$Classifications %>% distinct(classification_id_placeholder, table_id, .keep_all = TRUE)
  if(nrow(corpus_class) > 0) {
    message(paste("Generating embeddings for", nrow(corpus_class), "Classifications..."))
    corpus_class$unique_class_id <- paste(corpus_class$table_id, corpus_class$classification_id_placeholder, sep = "_")
    embeddings_list$Classifications <- model$encode(corpus_class$classification_desc, show_progress_bar = TRUE)
    rownames(embeddings_list$Classifications) <- corpus_class$unique_class_id
  }
  
  # --- 4. [NEW IN V9.0] Embed Categories ---
  corpus_categories <- db_raw$Categories %>%
    filter(!is.na(category_desc), category_desc != "") %>%
    distinct(category_id, classification_id_placeholder, table_id, .keep_all = TRUE)
  
  if(nrow(corpus_categories) > 0) {
    message(paste("Generating embeddings for", nrow(corpus_categories), "Categories..."))
    # Create a unique ID for each category instance
    corpus_categories$unique_cat_id <- paste(corpus_categories$table_id,
                                             corpus_categories$classification_id_placeholder,
                                             corpus_categories$category_id, sep = "_")
    embeddings_list$Categories <- model$encode(corpus_categories$category_desc, show_progress_bar = TRUE)
    rownames(embeddings_list$Categories) <- corpus_categories$unique_cat_id
  }
  
  
  semantic_database <- list(
    db = db_raw,
    embeddings = embeddings_list # The list now containing four embedding matrices
  )
  
  message(paste("Saving new CATEGORY-LEVEL semantic database to:", semantic_db_path))
  saveRDS(semantic_database, file = semantic_db_path)
  message("--- FOUR-LEVEL SEMANTIC DATABASE BUILD COMPLETE (V9.0) ---")
  return(invisible(semantic_database))
}


# =======================================================================
# --- SECTION 3: UNIFIED QUERY ENGINE (UPGRADED TO V9.0) ---
# =======================================================================

#' Queries the local SIDRA metadata database using semantic and/or keyword filters.
#' @param semantic_scope The component to search: "Tables", "Variables", "Classifications", or "Categories".
query_sidra <- function(mode = "hybrid",
                        semantic_phrase = NULL,
                        semantic_scope = "Tables", # Now accepts "Categories"
                        query_list = NULL,
                        filter_criteria = NULL,
                        top_n = 100,
                        semantic_db_path = SEMANTIC_DB_PATH, # Ensure this points to your new v9 path
                        model_name = "paraphrase-multilingual-mpnet-base-v2") {
  
  # --- Load Data ---
  if (!file.exists(semantic_db_path)) stop("Semantic DB not found for V9.0. Please run build_semantic_database().")
  semantic_db <- readRDS(semantic_db_path)
  db <- semantic_db$db
  
  # --- Helper: Semantic Search (UPGRADED for V9.0) ---
  .perform_semantic_search <- function(phrase, scope, n, model) {
    target_embeddings <- semantic_db$embeddings[[scope]]
    if (is.null(target_embeddings)) stop(paste("No embeddings found for scope:", scope))
    
    query_embedding <- model$encode(phrase)
    cosine_similarity <- function(a, B) { as.vector(crossprod(a, t(B))) / (as.vector(sqrt(crossprod(a))) * sqrt(rowSums(B^2))) }
    sim_scores <- cosine_similarity(query_embedding, target_embeddings)
    
    results <- tibble(item_id = rownames(target_embeddings), similarity = sim_scores) %>%
      arrange(desc(similarity)) %>% head(n)
    
    if (scope == "Tables") {
      return(results %>% mutate(table_id = as.integer(item_id)) %>% select(table_id, similarity))
    } else if (scope == "Variables") {
      var_map <- db$Variables %>% select(variable_id, table_id)
      return(results %>% mutate(item_id = as.integer(item_id)) %>% left_join(var_map, by = c("item_id" = "variable_id")) %>% group_by(table_id) %>% summarise(similarity = max(similarity)) %>% ungroup())
    } else if (scope == "Classifications") {
      class_map <- db$Classifications %>% mutate(unique_class_id = paste(table_id, classification_id_placeholder, sep = "_")) %>% select(unique_class_id, table_id)
      return(results %>% left_join(class_map, by = c("item_id" = "unique_class_id")) %>% group_by(table_id) %>% summarise(similarity = max(similarity)) %>% ungroup())
    } else if (scope == "Categories") {
      # [NEW IN V9.0] Map category search results back to their parent tables
      cat_map <- db$Categories %>%
        mutate(unique_cat_id = paste(table_id, classification_id_placeholder, category_id, sep = "_")) %>%
        select(unique_cat_id, table_id)
      return(results %>%
               left_join(cat_map, by = c("item_id" = "unique_cat_id")) %>%
               filter(!is.na(table_id)) %>% # Filter out any potential mismatches
               group_by(table_id) %>%
               summarise(similarity = max(similarity)) %>% # Find table with the most similar category
               ungroup())
    }
  }
  
  # --- Helper: Relational Filtering (Unchanged) ---
  .apply_relational_filters <- function(candidate_ids, query_list, filter_criteria) {
    if (length(candidate_ids) == 0) return(integer(0))
    filtered_ids <- candidate_ids
    if (!is.null(filter_criteria$min_munis)) {
      geo_ids <- db$Geographies %>% filter(table_id %in% filtered_ids, geo_level_api_code == "City") %>% mutate(num = readr::parse_number(geo_level_desc, locale = readr::locale(grouping_mark = "."))) %>% filter(num >= filter_criteria$min_munis) %>% pull(table_id) %>% unique()
      filtered_ids <- intersect(filtered_ids, geo_ids)
    }
    if (!is.null(filter_criteria$min_years) && !is.null(filter_criteria$year_range)) {
      time_ids <- db$Tables %>% filter(table_id %in% filtered_ids) %>% mutate(years = map(str_extract_all(period_raw, "[0-9]{4}"), as.integer), n_years = map_int(years, ~ sum(.x >= filter_criteria$year_range[1] & .x <= filter_criteria$year_range[2], na.rm = TRUE))) %>% filter(n_years >= filter_criteria$min_years) %>% pull(table_id) %>% unique()
      filtered_ids <- intersect(filtered_ids, time_ids)
    }
    if (!is.null(query_list)) {
      keyword_ids <- map(names(query_list), function(scope) {
        target_df <- db[[scope]]
        search_col_name <- names(target_df)[3]
        target_df %>% filter(table_id %in% filtered_ids, str_detect(str_to_lower(.data[[search_col_name]]), str_to_lower(query_list[[scope]]))) %>% pull(table_id) %>% unique()
      }) %>% Reduce(intersect, .)
      filtered_ids <- intersect(filtered_ids, keyword_ids)
    }
    return(filtered_ids)
  }
  
  # --- Main Logic (Unchanged) ---
  message(paste("--- Mode:", mode, "| Semantic Scope:", semantic_scope, "---"))
  SentenceTransformer <- reticulate::import("sentence_transformers")$SentenceTransformer
  model <- SentenceTransformer(model_name)
  
  semantic_candidates <- NULL
  final_ids <- NULL
  
  if (mode %in% c("hybrid", "semantic")) {
    if (is.null(semantic_phrase)) stop("Hybrid/Semantic mode requires a 'semantic_phrase'.")
    message("Stage 1: Performing semantic search...")
    semantic_candidates <- .perform_semantic_search(semantic_phrase, semantic_scope, top_n, model)
    final_ids <- semantic_candidates$table_id
    message(paste("  -> Found", length(final_ids), "semantically relevant candidate tables."))
  } else {
    final_ids <- db$Tables %>% filter(table_desc != "ERROR: FAILED_OR_TIMED_OUT") %>% pull(table_id)
  }
  
  if (mode %in% c("hybrid", "keyword")) {
    message("Stage 2: Applying relational filters...")
    final_ids <- .apply_relational_filters(final_ids, query_list, filter_criteria)
    message(paste("  -> Filtered down to", length(final_ids), "final tables."))
  }
  
  # --- Assemble Final, INFORMATIVE Results (Unchanged) ---
  if (length(final_ids) == 0) {
    message("\n>>> No tables found matching all specified criteria. <<<")
    return(tibble())
  }
  
  vars_agg <- db$Variables %>% group_by(table_id) %>% summarise(variables = paste(variable_desc_clean, collapse = " | "))
  class_agg <- db$Classifications %>% group_by(table_id) %>% summarise(classifications = paste(classification_desc, collapse = " | "))
  geo_agg <- db$Geographies %>% group_by(table_id) %>% summarise(geo_levels = paste(geo_level_api_code, collapse = ", "))
  
  results <- tibble(table_id = final_ids) %>%
    left_join(db$Tables, by = "table_id") %>%
    left_join(vars_agg, by = "table_id") %>%
    left_join(class_agg, by = "table_id") %>%
    left_join(geo_agg, by = "table_id")
  
  if (!is.null(semantic_candidates)) {
    results <- results %>%
      left_join(semantic_candidates, by = "table_id") %>%
      arrange(desc(similarity)) %>%
      select(table_id, similarity, table_desc, period_raw, variables, classifications, geo_levels)
  } else {
    results <- results %>%
      select(table_id, table_desc, period_raw, variables, classifications, geo_levels)
  }
  
  message(paste("\n>>> Found", nrow(results), "matching tables. <<<"))
  return(results)
}

# =======================================================================
# --- SECTION 4: EXAMPLE USAGE ---
# =======================================================================
# To run the script, uncomment the desired function calls below.

# --- BUILD STEPS (Run ONCE) ---
# message("--- Starting Step 1: Building Raw Metadata Database ---")
#build_sidra_database()
# message("--- Starting Step 2: Building Semantic Index ---")
build_semantic_database()

find_successor_by_variable <- query_sidra(
  mode = "hybrid",
  semantic_scope = "Variables",
  semantic_phrase = "Domicílios particulares permanentes",
  top_n = 50,
  filter_criteria = list(
    min_munis = 5000,
    min_years = 1,
    year_range = c(2000, 2025)
  )
)

print(find_successor_by_variable)


initial_candidate_tables <- query_sidra(
  mode = "keyword", # No semantic phrase, just filtering
  filter_criteria = list(
    min_munis = 5000,
    min_years = 2,
    year_range = c(2000, 2024)
  )
)

write.csv(initial_candidate_tables)

# --- V9.0 CATEGORY SEARCH EXAMPLE ---
# This query solves the problem of finding "Microcomputador" which failed with V8.0
# It will now search within the individual categories of all classifications.

find_computer_2022_v9 <- query_sidra(
  mode = "hybrid",
  semantic_scope = "Categories", # Use the new V9.0 capability
  semantic_phrase = "Microcomputador com acesso à internet",
  top_n = 100,
  filter_criteria = list(
    min_munis = 5000,
    min_years = 1,
    year_range = c(2011, 2024) # Strictly filter for the 2022 census
  )
)

write.csv(find_computer_2022_v9)


# Query 5: Let's take a step back and search for tables whose TITLE
# is semantically related to our goal. This is a broader but more robust search.
find_durable_goods_2022 <- query_sidra(
  mode = "hybrid",
  semantic_scope = "Tables", # Search the general table description
  semantic_phrase = "domicílios por acesso a bens e serviços", # A broader concept
  top_n = 20,
  filter_criteria = list(
    min_munis = 1000,
    min_years = 1,
    year_range = c(2011, 2024)
  )
)

write.csv(find_durable_goods_2022)





# Define the core themes for our demographic basis
search_themes <- c(
  "população por alfabetização e nível de instrução",
  "domicílios por saneamento básico e esgotamento sanitário",
  "domicílios por acesso a bens e internet",
  "população por condição de atividade e trabalho",
  "população por classes de rendimento mensal",
  "condições de moradia e infraestrutura domiciliar",
  "nupcialidade, casamentos e divórcios"
)

# Loop through each theme and perform a broad, filtered search
for (theme in search_themes) {
  
  cat("\n=====================================================\n")
  cat("SEARCHING FOR THEME:", theme, "\n")
  cat("=====================================================\n\n")
  
  # Use a tryCatch to ensure the loop continues even if one query fails
  tryCatch({
    
    candidate_tables <- query_sidra(
      mode = "hybrid",
      semantic_scope = "Tables",
      semantic_phrase = theme,
      top_n = 20, # Keep the net reasonably wide for each theme
      filter_criteria = list(
        min_munis = 5000,
        min_years = 1,
        year_range = c(2000, 2024)
      )
    )
    
    # Use write.csv for direct, complete console output
    if (nrow(candidate_tables) > 0) {
      write.csv(candidate_tables)
    } else {
      cat(">>> No tables found matching the criteria for this theme.\n")
    }
    
  }, error = function(e) {
    cat("ERROR during query for theme '", theme, "': ", e$message, "\n")
  })
}