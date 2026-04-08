import time
import random
from rich.console import Console
from rich.table import Table
from rich import box
from rich.live import Live

def generate_table() -> Table:
    """Creates a new table with updated data"""
    table = Table(
        title="[bold cyan]ULTIMATE SCALPING TOOL (LIVE)[/bold cyan]", 
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True, 
        header_style="bold white"
    )

    table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
    table.add_column("Status / Value", justify="center", style="bold")

    # Let's generate some random fluctuating live values
    rsi = round(random.uniform(30.0, 70.0), 2)
    vwap = round(random.uniform(4600.0, 4700.0), 2)
    adx = round(random.uniform(15.0, 25.0), 2)
    
    # Change colors dynamically based on the numbers!
    if rsi < 40:
        rsi_style = "bold white on green"
    elif rsi > 60:
        rsi_style = "bold white on red"
    else:
        rsi_style = "black on bright_yellow"
        
    table.add_row("Trend (TF)", "[bright_black]Sideways[/bright_black]")
    table.add_row("Momentum (TF)", "[bright_black]Neutral[/bright_black]")
    table.add_row("Volume (CMF)", "[bright_black]Neutral[/bright_black]")
    table.add_row("Basic Signal", "[bright_black]No Trade[/bright_black]")
    table.add_row("Advanced Signal", "[black on bright_yellow] No Trade [/]")
    table.add_row("RSI", f"[{rsi_style}] {rsi} [/]")
    table.add_row("HTF Filter", "[bold white on red] Bearish [/]")
    table.add_row("VWAP", f"[bold white on green] {vwap} [/]")
    table.add_row("ADX", f"[bold white on red] {adx} [/]")
    table.add_row("Mode", "[bold white on green] Custom [/]")
    table.add_row("Regime", "[bold white on green] Low Volatility [/]")
    
    return table

def run_live_dashboard():
    console = Console()
    
    console.print("\n[bold italic magenta]Starting live dashboard simulation...[/bold italic magenta]")
    
    # The Live context manager automatically handles replacing the previous table in the terminal!
    with Live(generate_table(), refresh_per_second=4, console=console) as live:
        try:
            while True: # Run continuously
                time.sleep(1.0) # pause for 1s between updates
                live.update(generate_table()) # push the new data to the terminal
        except KeyboardInterrupt:
            pass # clean exit when you press Ctrl+C
            
    console.print("\n[bold green]Live updating demo finished![/bold green]")

if __name__ == "__main__":
    run_live_dashboard()
