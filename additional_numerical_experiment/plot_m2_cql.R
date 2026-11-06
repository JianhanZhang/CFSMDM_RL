library(ggplot2)
library(tidyverse)
library(viridis)
library(ggpubr)
library(latex2exp)
library(patchwork)

root_folder <- "" # Please fill in your root folder

file_z <- c(
  file.path(root_folder, "/results/results_m2_Z_cql.csv")
)

file_n <- c(
  file.path(root_folder, "/results/results_m2_N_cql.csv")
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

plot_N <- function(files) {
  plots <- list()
  for (file in files) {
    env_setting <- strsplit(file, "_")[[1]][5]
    fqi_setting <- strsplit(file, "_")[[1]][16]
    raw_dat <- read_csv(file)
    filename_prefix <- paste(env_setting, fqi_setting, sep = "_")
    df <- raw_dat 
    df <- df %>%
      group_by(N, T, method) %>%
      summarise(across(c(cf, reward), list(
        mean = mean, lb = ~ quantile(., probs = 0.025), ub = ~ quantile(., probs = 0.975)
      )))
    df <- df %>%
      filter(!method %in% excluded_list) %>%
      mutate(method = recode(method, !!!names_map))
    print(df)
    
    # plot cf
    p_cf <- ggplot(aes(
      x = N,
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
      scale_x_continuous(breaks = c(100, 500, 1000, 2000)) +
      scale_shape_manual(values = my_shapes) +
      scale_x_log10(breaks = c(100, 200, 500, 1000, 2000)) + 
      theme(legend.position = "bottom")
    ggsave(file.path(root_folder, "/plots/N_CF_plot_m2_cql.png"),
           p_cf,
           scale = 1, width = 5, height = 6
    )
    
    # plot reward
    p_reward <- ggplot(aes(
      x = N,
      y = reward_mean,
      color = method,
      shape = method
    ), data = df) +
      geom_point(size = 4, stroke=2) +
      geom_line(size = 1.2) +
      geom_ribbon(aes(ymin = reward_lb, ymax = reward_ub, fill = method),
                  alpha = 0.2,
                  color = NA
      ) +
      theme_bw() +
      labs(y = "Cumulative discounted reward") +
      scale_shape_manual(values = my_shapes) + 
      scale_x_log10(breaks = c(100, 200, 500, 1000, 2000)) + 
      theme(legend.position = "bottom")
    ggsave(file.path(root_folder, "/plots/N_reward_plot_m2_cql.png"),
           p_reward,
           scale = 1, width = 5, height = 6
    )
    
    # plot cf vs reward
    p_cf_reward <- ggplot(aes(
      x = cf_mean,
      y = reward_mean,
      color = method,
      shape = method,
      alpha = N,
    ), data = df) +
      geom_point(size = 4, stroke=2) +
      theme_bw() +
      labs(x = "CF metric", y = "Cumulative discounted reward") +
      scale_alpha_continuous(range = c(0.3, 1), breaks = c(100, 200, 500, 1000, 2000)) +
      guides(colour = guide_legend(nrow = 1), alpha = guide_legend(nrow = 1)) +
      theme(legend.position = "bottom") +
      scale_shape_manual(values = my_shapes) +
      guides(colour = guide_legend(nrow = 2), alpha = guide_legend(nrow = 2), shape = guide_legend(nrow = 2))
    ggsave(file.path(root_folder, "/plots/N_CF_reward_plot_m2_cql.png"),
           p_cf_reward,
           scale = 1, width = 5, height = 6
    )

    plots <- list(p_cf, p_reward, p_cf_reward)
  }
  return(plots)
}

plot_Z <- function(files) {
  plots <- list()
  for (file in files) {
    env_setting <- strsplit(file, "_")[[1]][5]
    fqi_setting <- strsplit(file, "_")[[1]][16]
    raw_dat <- read_csv(file)
    filename_prefix <- paste(env_setting, fqi_setting, sep = "_")
    df <- raw_dat %>%
      group_by(zcoef, N, T, method) %>%
      summarise(across(c(cf, reward), list(
        mean = mean, lb = ~ quantile(., probs = 0.025, na.rm = TRUE), ub = ~ quantile(., probs = 0.975, na.rm = TRUE) # modified to rmv na
      )))
    df <- df %>%
      filter(!method %in% excluded_list) %>%
      mutate(method = recode(method, !!!names_map))

    # plot cf
    p_cf <- ggplot(aes(
      x = zcoef,
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
      labs(y = "CF metric", x = expression(delta)) +
      scale_shape_manual(values = my_shapes) + 
      theme(legend.position = "bottom")
    ggsave(file.path(root_folder, "/plots/Z_CF_plot_m2_cql.png"),
      p_cf,
      scale = 1, width = 5, height = 6
    )

    # plot reward
    p_reward <- ggplot(aes(
      x = zcoef,
      y = reward_mean,
      color = method,
      shape = method
    ), data = df) +
      geom_point(size = 4, stroke=2) +
      geom_line(size = 1.2) +
      geom_ribbon(aes(ymin = reward_lb, ymax = reward_ub, fill = method),
        alpha = 0.2,
        color = NA
      ) +
      theme_bw() +
      labs(y = "Cumulative discounted reward") +
      scale_shape_manual(values = my_shapes)
    ggsave(file.path(root_folder, "/plots/Z_reward_plot_m2_cql.png"),
      p_reward,
      scale = 1, width = 5, height = 6
    )

    # plot cf vs reward
    p_cf_reward <- ggplot(aes(
      x = cf_mean,
      y = reward_mean,
      color = zcoef,
      shape = method
    ), data = df) +
      geom_point(size = 4, stroke=2) +
      theme_bw() +
      labs(x = "CF metric", y = "Cumulative discounted reward") +
      scale_color_viridis(name = "N") +
      scale_shape_manual(values = my_shapes) +
      guides(colour = guide_legend(nrow = 1), alpha = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
    ggsave(file.path(root_folder, "/plots/Z_CF_reward_plot_m2_cql.png"),
      p_cf_reward,
      scale = 1, width = 5, height = 6
    )
    plots <- list(p_cf, p_reward, p_cf_reward)
  }
  return(plots)
}

ps_z <- plot_Z(file_z)
ps_n <- plot_N(file_n)

# Only keep one of the two "Method" legends from the patchwork
ps_n[[3]] <- ps_n[[3]] + guides(color = "none", shape = "none")

combined_plot <- ps_n[[1]] + ps_n[[2]] + ps_n[[3]] + ps_z[[1]] + plot_layout(ncol = 4, guides="collect") + 
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

ggsave(combined_plot, file = file.path(root_folder, "/plots/combined_plot_m2_cql.png"), scale = 1, width = 20, height = 8)
