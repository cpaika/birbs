use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, List, ListItem, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub total_epochs: usize,
    pub train_loss: f32,
    pub test_loss: f32,
    pub test_accuracy: f32,
    pub batch: usize,
    pub total_batches: usize,
    pub current_batch_loss: Option<f32>,
}

pub struct TrainingUI {
    train_loss_history: Vec<(f64, f64)>,
    test_loss_history: Vec<(f64, f64)>,
    accuracy_history: Vec<(f64, f64)>,
    log_messages: Vec<String>,
    current_metrics: Option<TrainingMetrics>,
    start_time: Instant,
    max_logs: usize,
}

impl TrainingUI {
    pub fn new() -> Self {
        Self {
            train_loss_history: Vec::new(),
            test_loss_history: Vec::new(),
            accuracy_history: Vec::new(),
            log_messages: Vec::new(),
            current_metrics: None,
            start_time: Instant::now(),
            max_logs: 100,
        }
    }

    pub fn add_metrics(&mut self, metrics: TrainingMetrics) {
        let epoch = metrics.epoch as f64;

        // Add to history
        self.train_loss_history.push((epoch, metrics.train_loss as f64));
        self.test_loss_history.push((epoch, metrics.test_loss as f64));
        self.accuracy_history.push((epoch, metrics.test_accuracy as f64 * 100.0));

        // Add log message
        let elapsed = self.start_time.elapsed();
        let log_msg = format!(
            "[{:02}:{:02}:{:02}] Epoch {}/{} | Train Loss: {:.4} | Test Loss: {:.4} | Accuracy: {:.2}%",
            elapsed.as_secs() / 3600,
            (elapsed.as_secs() % 3600) / 60,
            elapsed.as_secs() % 60,
            metrics.epoch,
            metrics.total_epochs,
            metrics.train_loss,
            metrics.test_loss,
            metrics.test_accuracy * 100.0
        );
        self.log_messages.push(log_msg);

        // Keep only the last max_logs messages
        if self.log_messages.len() > self.max_logs {
            self.log_messages.remove(0);
        }

        self.current_metrics = Some(metrics);
    }

    pub fn add_log(&mut self, message: String) {
        let elapsed = self.start_time.elapsed();
        let log_msg = format!(
            "[{:02}:{:02}:{:02}] {}",
            elapsed.as_secs() / 3600,
            (elapsed.as_secs() % 3600) / 60,
            elapsed.as_secs() % 60,
            message
        );
        self.log_messages.push(log_msg);

        if self.log_messages.len() > self.max_logs {
            self.log_messages.remove(0);
        }
    }

    pub fn update_batch_progress(&mut self, batch: usize, total_batches: usize, loss: f32) {
        if let Some(ref mut metrics) = self.current_metrics {
            metrics.batch = batch;
            metrics.total_batches = total_batches;
            metrics.current_batch_loss = Some(loss);
        }
    }

    fn render(&self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(frame.area());

        // Left side: Charts
        self.render_charts(frame, chunks[0]);

        // Right side: Logs and progress
        self.render_logs(frame, chunks[1]);
    }

    fn render_charts(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Loss chart
        self.render_loss_chart(frame, chunks[0]);

        // Accuracy chart
        self.render_accuracy_chart(frame, chunks[1]);
    }

    fn render_loss_chart(&self, frame: &mut Frame, area: Rect) {
        if self.train_loss_history.is_empty() {
            let block = Block::default()
                .title("Training & Test Loss")
                .borders(Borders::ALL);
            frame.render_widget(block, area);
            return;
        }

        let max_epoch = self.train_loss_history.last().map(|(e, _)| *e).unwrap_or(1.0);
        let max_loss = self
            .train_loss_history
            .iter()
            .chain(self.test_loss_history.iter())
            .map(|(_, l)| *l)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_loss = self
            .train_loss_history
            .iter()
            .chain(self.test_loss_history.iter())
            .map(|(_, l)| *l)
            .fold(f64::INFINITY, f64::min);

        let datasets = vec![
            Dataset::default()
                .name("Train Loss")
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::Cyan))
                .data(&self.train_loss_history),
            Dataset::default()
                .name("Test Loss")
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::Yellow))
                .data(&self.test_loss_history),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .title("Training & Test Loss")
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    .title("Epoch")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, max_epoch.max(1.0)])
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw(format!("{:.0}", max_epoch / 2.0)),
                        Span::raw(format!("{:.0}", max_epoch)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title("Loss")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_loss * 0.9, max_loss * 1.1])
                    .labels(vec![
                        Span::raw(format!("{:.2}", min_loss)),
                        Span::raw(format!("{:.2}", (min_loss + max_loss) / 2.0)),
                        Span::raw(format!("{:.2}", max_loss)),
                    ]),
            );

        frame.render_widget(chart, area);
    }

    fn render_accuracy_chart(&self, frame: &mut Frame, area: Rect) {
        if self.accuracy_history.is_empty() {
            let block = Block::default()
                .title("Test Accuracy")
                .borders(Borders::ALL);
            frame.render_widget(block, area);
            return;
        }

        let max_epoch = self.accuracy_history.last().map(|(e, _)| *e).unwrap_or(1.0);

        let datasets = vec![Dataset::default()
            .name("Accuracy")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Green))
            .data(&self.accuracy_history)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .title("Test Accuracy (%)")
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    .title("Epoch")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, max_epoch.max(1.0)])
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw(format!("{:.0}", max_epoch / 2.0)),
                        Span::raw(format!("{:.0}", max_epoch)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title("Accuracy")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, 100.0])
                    .labels(vec![
                        Span::raw("0%"),
                        Span::raw("50%"),
                        Span::raw("100%"),
                    ]),
            );

        frame.render_widget(chart, area);
    }

    fn render_logs(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(12), Constraint::Min(0)])
            .split(area);

        // Current status
        self.render_status(frame, chunks[0]);

        // Log messages
        self.render_log_messages(frame, chunks[1]);
    }

    fn render_status(&self, frame: &mut Frame, area: Rect) {
        let elapsed = self.start_time.elapsed();
        let hours = elapsed.as_secs() / 3600;
        let minutes = (elapsed.as_secs() % 3600) / 60;
        let seconds = elapsed.as_secs() % 60;

        let status_text = if let Some(ref metrics) = self.current_metrics {
            let progress = (metrics.epoch as f32 / metrics.total_epochs as f32) * 100.0;
            let batch_info = if let Some(batch_loss) = metrics.current_batch_loss {
                format!(
                    "Batch {}/{} | Loss: {:.4}",
                    metrics.batch, metrics.total_batches, batch_loss
                )
            } else {
                String::from("Waiting for batch data...")
            };

            vec![
                Line::from(vec![
                    Span::styled("Epoch: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{}/{}", metrics.epoch, metrics.total_epochs),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" ({:.1}%)", progress),
                        Style::default().fg(Color::Gray),
                    ),
                ]),
                Line::from(Span::raw(batch_info)),
                Line::from(vec![
                    Span::styled("Train Loss: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{:.4}", metrics.train_loss),
                        Style::default().fg(Color::Cyan),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Test Loss: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{:.4}", metrics.test_loss),
                        Style::default().fg(Color::Yellow),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Test Accuracy: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{:.2}%", metrics.test_accuracy * 100.0),
                        Style::default().fg(Color::Green),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Elapsed: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{:02}:{:02}:{:02}", hours, minutes, seconds),
                        Style::default().fg(Color::Magenta),
                    ),
                ]),
            ]
        } else {
            // During initialization, show recent log messages
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Initializing training...", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(vec![
                    Span::styled("Elapsed: ", Style::default().fg(Color::White)),
                    Span::styled(
                        format!("{:02}:{:02}:{:02}", hours, minutes, seconds),
                        Style::default().fg(Color::Magenta),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Recent activity:", Style::default().fg(Color::Cyan)),
                ]),
            ];

            // Add last 3 log messages
            let recent_logs = self.log_messages.iter().rev().take(3).rev();
            for log in recent_logs {
                lines.push(Line::from(log.as_str()));
            }

            lines
        };

        let status = Paragraph::new(status_text).block(
            Block::default()
                .title("Training Status")
                .borders(Borders::ALL),
        );

        frame.render_widget(status, area);
    }

    fn render_log_messages(&self, frame: &mut Frame, area: Rect) {
        let logs: Vec<ListItem> = self
            .log_messages
            .iter()
            .rev()
            .map(|msg| ListItem::new(msg.as_str()))
            .collect();

        let logs_list = List::new(logs).block(
            Block::default()
                .title("Training Log")
                .borders(Borders::ALL),
        );

        frame.render_widget(logs_list, area);
    }
}

pub fn run_training_ui<F>(training_fn: F) -> Result<()>
where
    F: FnOnce(Arc<Mutex<TrainingUI>>) -> Result<()> + Send + 'static,
{
    // Setup terminal
    if let Err(e) = enable_raw_mode() {
    }

    let mut stdout = io::stdout();
    if let Err(e) = execute!(stdout, EnterAlternateScreen) {
        return Err(e.into());
    }

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let ui = Arc::new(Mutex::new(TrainingUI::new()));
    let ui_clone = ui.clone();

    // Spawn training thread
    let training_handle = std::thread::spawn(move || training_fn(ui_clone));

    // UI rendering loop
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();
    let mut event_polling_disabled = false;

    loop {
        terminal.draw(|f| {
            let ui = ui.lock().unwrap();
            ui.render(f);
        })?;

        // Check for keyboard events, but don't fail if stdin isn't available
        if !event_polling_disabled {
            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            match event::poll(timeout) {
                Ok(true) => {
                    // Event available - try to read it
                    match event::read() {
                        Ok(Event::Key(key)) => {
                            if key.kind == KeyEventKind::Press {
                                match key.code {
                                    KeyCode::Char('q') | KeyCode::Esc => {
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Ok(_) => {} // Other events, ignore
                        Err(e) => {
                            // Failed to read event, but continue
                        }
                    }
                }
                Ok(false) => {} // No event available
                Err(e) => {
                    // Polling failed - likely stdin isn't available
                    // Just continue without keyboard input support
                    event_polling_disabled = true;
                }
            }
        } else {
            // Event polling is disabled, just sleep
            std::thread::sleep(tick_rate);
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }

        // Check if training is done
        if training_handle.is_finished() {
            std::thread::sleep(Duration::from_secs(2)); // Show final state for 2 seconds
            break;
        }
    }

    // Cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Wait for training thread to complete
    training_handle.join().unwrap()?;

    Ok(())
}
