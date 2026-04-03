import os

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class Colors:
    RED = '\033[91m'     
    YELLOW = '\033[93m'  
    WHITE = '\033[97m'   
    GREEN = '\033[92m'   
    CYAN = '\033[96m'    
    RESET = '\033[0m'

def format_time(seconds):
    if seconds <= 0: return "0:00.000"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"

def get_board_string(env, highlight_teams=None):
    """Generates the telemetry board. Highlighted teams will appear in Yellow."""
    if highlight_teams is None:
        highlight_teams = []

    lines = []
    lines.append(f"{Colors.CYAN}========================================================================================{Colors.RESET}")
    lines.append(f"  FIA FORMULA MAR1 WORLD CHAMPIONSHIP - LAP {env.current_lap:2d} / {env.total_laps}  |  TRACK: GREEN  |  CARS: {len(env.cars)}")
    lines.append(f"{Colors.CYAN}========================================================================================{Colors.RESET}")
    lines.append(" P  | DRIVER | GAP        | INT        | LAST LAP | TYRE | LAPS | BATTERY | STP | MODE")
    lines.append("----------------------------------------------------------------------------------------")
    
    leader_time = env.cars[0]["total_race_time"] if len(env.cars) > 0 else 0
    
    for i, car in enumerate(env.cars):
        # Determine if this row should be highlighted (AI teams)
        is_highlighted = car['team'] in highlight_teams
        row_color = Colors.YELLOW if is_highlighted else ""
        reset_color = Colors.RESET if is_highlighted else ""

        pos = f"{i+1:2d}"
        driver = f"{car['id']:<6}"
        
        if i == 0:
            gap, interval = "Leader    ", "-         "
        else:
            gap = f"+{car['total_race_time'] - leader_time:<9.3f}"
            interval = f"+{car['total_race_time'] - env.cars[i-1]['total_race_time']:<9.3f}"
            
        t_comp = car["tyre_compound"]
        tyre_str = f"{Colors.RED}S{Colors.RESET}" if t_comp == 1 else f"{Colors.YELLOW}M{Colors.RESET}" if t_comp == 2 else f"{Colors.WHITE}H{Colors.RESET}"
            
        ers_blocks = int((car["battery"] / 1.0) * 5)
        ers_bar = f"[{'|'*ers_blocks}{' '*(5-ers_blocks)}]"
        
        status = car['status']
        if status == "OVR":
            status = f"{Colors.GREEN}{status}{Colors.RESET}"
            
        # Apply the highlight color to the Position and Driver columns
        row = f" {row_color}{pos}{reset_color} | {row_color}{driver}{reset_color} | {gap} | {interval} | {format_time(car['last_lap_time'])} |  {tyre_str}   |  {int(car['tyre_age']):2d}  | {ers_bar} |  {car['pit_stops']}  | {status}"
        lines.append(row)
        
    lines.append(f"{Colors.CYAN}========================================================================================{Colors.RESET}")
    lines.append(f"                                                            Simulation by Jan A. Krzywda")
    return "\n".join(lines)

def render_telemetry(env, highlight_teams=None):
    """Clears the terminal and prints the live telemetry with optional highlights."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(get_board_string(env, highlight_teams))

def get_monospace_font(size=14):
    if not PIL_AVAILABLE: return None
    fallbacks = ["consola.ttf", "cour.ttf", "lucon.ttf", "Menlo.ttc", "Monaco.ttf", "DejaVuSansMono.ttf"]
    for font_name in fallbacks:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    return ImageFont.load_default()

def draw_ansi_text_to_image(text, font):
    img = Image.new('RGB', (820, 580), color=(15, 15, 18)) 
    draw = ImageDraw.Draw(img)
    color_map = {
        '91m': (255, 85, 85),   # Red
        '93m': (255, 235, 85),  # Yellow
        '97m': (255, 255, 255), # White
        '92m': (85, 255, 85),   # Green
        '96m': (85, 255, 255),  # Cyan
        '0m':  (200, 200, 200)  # Reset (Light Gray)
    }
    char_width = font.getlength("A") if hasattr(font, 'getlength') else font.getbbox("A")[2] if hasattr(font, 'getbbox') else 8
    
    current_y = 15
    for line in text.split('\n'):
        current_x = 15
        current_color = color_map['0m']
        parts = line.split('\033[')
        for i, part in enumerate(parts):
            if i == 0:
                draw.text((current_x, current_y), part, fill=current_color, font=font)
                current_x += len(part) * char_width
            else:
                code_end = part.find('m')
                if code_end != -1:
                    code = part[:code_end+1]
                    text_part = part[code_end+1:]
                    if code in color_map:
                        current_color = color_map[code]
                    draw.text((current_x, current_y), text_part, fill=current_color, font=font)
                    current_x += len(text_part) * char_width
        current_y += 18 
    return img