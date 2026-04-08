from rich.console import Console
from rich.table import Table
from rich import box

def draw_scalping_table():
    console = Console()

    # Create a table with a bold header and specific box style
    table = Table(
        title="[bold cyan]ULTIMATE SCALPING TOOL[/bold cyan]", 
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True, 
        header_style="bold white"
    )

    # Add columns
    table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
    table.add_column("Status / Value", justify="center", style="bold")

    # Add rows with specific color formatting to match your image
    table.add_row("Trend (TF)", "[bright_black]Sideways[/bright_black]")
    table.add_row("Momentum (TF)", "[bright_black]Neutral[/bright_black]")
    table.add_row("Volume (CMF)", "[bright_black]Neutral[/bright_black]")
    table.add_row("Basic Signal", "[bright_black]No Trade[/bright_black]")
    
    # Yellow warning colors
    table.add_row("Advanced Signal", "[black on bright_yellow] No Trade [/]")
    table.add_row("RSI", "[black on bright_yellow] 47.06 [/]")
    
    # Red/Bearish colors
    table.add_row("HTF Filter", "[bold white on red] Bearish [/]")
    
    # Green/Bullish/Value colors
    table.add_row("VWAP", "[bold white on green] 4659.83 [/]")
    table.add_row("ADX", "[bold white on red] 16.25 [/]")
    table.add_row("Mode", "[bold white on green] Custom [/]")
    table.add_row("Regime", "[bold white on green] Low Volatility [/]")

    # Print the table to the terminal
    console.print(table)

if __name__ == "__main__":
    draw_scalping_table()
