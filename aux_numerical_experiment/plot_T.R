library(ggplot2)
library(tidyverse)
library(viridis)
library(ggpubr)
library(latex2exp)
library(patchwork)

root_folder <- "" # Please fill in your root folder

file_m1 <- c(
  file.path(root_folder, "/results_aux/results_m1_T.csv")
)

file_m2 <- c(
  file.path(root_folder, "/results_aux/results_m2_T.csv")
)

names_map <- list(
  "Full" = "full",
  "Unaware" = "Unaware",
  "Random" = "Random", 
  "CFSMDM" = "cfsmdm",
  "CFSDP" = "cfsdp", 
  "FLAP" = "flap_m", 
  "ECOCF" = "ecocf_m"
)
excluded_list <- c("Ours_reward_cf10")
my_shapes <- c(2, 3, 4, 5, 7, 8, 9) 

plot_m1 <- function(files) {
  plots <- list()
  for (file in files) {
    raw_dat <- read_csv(file)
    df <- raw_dat #%>% dplyr::select(-c(cf)) %>% rename(cf = cf)
    df <- df %>%
      group_by(T, method) %>%
      summarise(across(c(cf), list(
        mean = mean, lb = ~ quantile(., probs = 0.025), ub = ~ quantile(., probs = 0.975)
      )))
    df <- df %>%
      filter(!method %in% excluded_list) %>%
      mutate(method = recode(method, !!!names_map))
    print(df)
    
    # plot m1
    p_m1 <- ggplot(aes(
      x = T,
      y = cf_mean,
      color = method,
      shape = method
    ), data = df) +
      geom_point(size = 4, stroke=2) +
      geom_line(size = 1.2) +
      geom_ribbon(aes(ymin = cf_lb, ymax = cf_ub, fill = method),
                  alpha = 0.2,
                  color = NA
      ) +
      theme_bw() +
      labs(y = "CF metric") +
      #scale_x_continuous(breaks = c(100, 500, 1000, 2000)) +
      scale_shape_manual(values = my_shapes) +
      scale_x_log10(breaks = c(10, 20, 30, 50, 100)) + 
      theme(legend.position = "bottom")
    ggsave(file.path(root_folder, "/plots_aux/T_plot_m1.png"),
           p_m1,
           scale = 1, width = 5, height = 6
    )

    plots <- list(p_m1)
  }
  return(plots)
}

plot_m2 <- function(files) {
  plots <- list()
  for (file in files) {
    raw_dat <- read_csv(file)
    df <- raw_dat #%>% dplyr::select(-c(cf)) %>% rename(cf = cf)
    df <- df %>%
      group_by(T, method) %>%
      summarise(across(c(cf), list(
        mean = mean, lb = ~ quantile(., probs = 0.025), ub = ~ quantile(., probs = 0.975)
      )))
    df <- df %>%
      filter(!method %in% excluded_list) %>%
      mutate(method = recode(method, !!!names_map))
    print(df)
    
    # plot m2
    p_m2 <- ggplot(aes(
      x = T,
      y = cf_mean,
      color = method,
      shape = method
    ), data = df) +
      geom_point(size = 4, stroke=2) +
      geom_line(size = 1.2) +
      geom_ribbon(aes(ymin = cf_lb, ymax = cf_ub, fill = method),
                  alpha = 0.2,
                  color = NA
      ) +
      theme_bw() +
      labs(y = "CF metric") +
      scale_shape_manual(values = my_shapes) +
      scale_x_log10(breaks = c(10, 20, 30, 50, 100)) + 
      theme(legend.position = "bottom")
    ggsave(file.path(root_folder, "/plots_aux/T_plot_m2.png"),
           p_m2,
           scale = 1, width = 5, height = 6
    )

    plots <- list(p_m2)
  }
  return(plots)
}


ps_m1 <- plot_m1(file_m1)
ps_m2 <- plot_m2(file_m2)

# Only keep one of the two "Method" legends from the patchwork
combined_plot <- ps_m1[[1]] + ps_m2[[1]] + plot_layout(ncol = 2, guides="collect") + 
plot_annotation(tag_levels = 'a') & theme(
  legend.position = "bottom",
  legend.title = element_text(size = 20, face = "bold"), 
  legend.text = element_text(size = 20),                
  legend.key.size = unit(1.5, "cm"), 
  axis.text = element_text(size = 20, face = "bold", color = "black"),
  axis.title = element_text(size = 20, face = "bold"),
  panel.border = element_rect(linewidth = 1.5, color = "black"),
  axis.ticks = element_line(linewidth = 1),
  plot.tag = element_text(size = 20, face = "bold")
)

# View it
combined_plot

ggsave(combined_plot, file = file.path(root_folder, "/plots_aux/combined_plot_T_m1_m2.png"), scale = 1, width = 16, height = 8)
