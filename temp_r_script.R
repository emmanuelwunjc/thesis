
        load("data/hints7_public copy.rda")
        data_name <- ls()[1]
        data <- get(data_name)
        
        # Convert to data frame and save as CSV
        write.csv(data, "temp_hints_data.csv", row.names=FALSE)
        
        # Print some info
        cat("Data shape:", nrow(data), "x", ncol(data), "\n")
        