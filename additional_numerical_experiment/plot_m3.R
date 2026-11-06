library(ggplot2)
library(tidyverse)
library(viridis)
library(ggpubr)
library(latex2exp)
library(patchwork)

root_folder <- "" # Please fill in your root folder

file_m3 <- c(
  file.path(root_folder, "/results/results_m3_N.csv")
)

names_map <- list(
  "Full" = "full",
  "Unaware" = "Unaware",
  "Random" = "Random", 
  "CFSMDM_QRNN" = "cfsmdm_qrnn",
  "CFSDP" = "cfsdp", 
  "FLAP" = "flap_m", 
  "ECOCF" = "ecocf_m"
)
my_shapes <- c(2, 3, 4, 5, 7, 8, 9) 
excluded_list <- c("")

plot_m3 <- function(files) {
  plots <- list()
  for (file in files) {
    raw_dat <- read_csv(file)
    df <- raw_dat 
    df <- df %>%
      group_by(N, method) %>%
      summarise(across(c(cf, reward), list(
        mean = mean, lb = ~ quantile(., probs = 0.025), ub = ~ quantile(., probs = 0.975)
      )))
    df <- df %>%
      filter(!method %in% excluded_list) %>%
      mutate(method = recode(method, !!!names_map))
    print(df)
    
    # plot m3
    p_m3_cf <- ggplot(aes(
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
      labs(x = "N") +
      scale_x_continuous(breaks = c(500, 1000, 2000, 3000, 5000)) + 
      scale_shape_manual(values = my_shapes) +
      theme(legend.position = "bottom")
    ggsave(file.path(root_folder, "/plots/plot_m3_N_cf.png"),
           p_m3_cf,
           scale = 1, width = 5, height = 6
    )

    p_m3_reward <- ggplot(aes(
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
      labs(y = "Policy Value") +
      labs(x = "N") +
      scale_x_continuous(breaks = c(500, 1000, 2000, 3000, 5000)) + 
      scale_shape_manual(values = my_shapes) +
      theme(legend.position = "bottom")
    ggsave(file.path(root_folder, "/plots/plot_m3_N_reward.png"),
           p_m3_reward,
           scale = 1, width = 5, height = 6
    )

    plots <- list(p_m3_cf, p_m3_reward)
  }
  return(plots)
}

ps_m3 <- plot_m3(file_m3)

# Only keep one of the two "Method" legends from the patchwork
combined_plot <- ps_m3[[1]] + ps_m3[[2]] + plot_layout(ncol = 2, guides="collect") + 
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

ggsave(combined_plot, file = file.path(root_folder, "/plots/combined_plot_m3.png"), scale = 1, width = 24, height = 10)
