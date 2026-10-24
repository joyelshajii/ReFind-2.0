using System.Diagnostics;
using System.ComponentModel;
using System.Drawing.Drawing2D;
using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace ReFind.Tray;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        using var mutex = new Mutex(true, @"Local\ReFind.Tray", out var isFirstInstance);
        if (!isFirstInstance) return;

        ApplicationConfiguration.Initialize();
        Application.Run(new TrayContext());
    }
}

internal sealed class TrayContext : ApplicationContext
{
    private readonly NotifyIcon _tray;
    private readonly HotkeyWindow _hotkey;
    private readonly SearchOverlay _overlay;
    private readonly BackendHost _backend;
    private readonly System.Windows.Forms.Timer _healthTimer;

    public TrayContext()
    {
        var root = FindProjectRoot();
        _backend = new BackendHost(root);
        _overlay = new SearchOverlay(_backend);
        _hotkey = new HotkeyWindow(() => {
            if (_overlay.Visible) _overlay.HideOverlay();
            else _overlay.ShowOverlay();
        });

        var menu = new ContextMenuStrip();
        menu.Items.Add("Search files    Alt+Space", null, (_, _) => _overlay.ShowOverlay());
        menu.Items.Add("Open ReFind in browser", null, async (_, _) => await OpenBrowserAsync(root));
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("Exit ReFind", null, (_, _) => ExitThread());

        _tray = new NotifyIcon {
            Icon = CreateTrayIcon(),
            Text = "ReFind — Alt+Space to search",
            ContextMenuStrip = menu,
            Visible = true,
        };
        _tray.DoubleClick += (_, _) => _overlay.ShowOverlay();

        if (!_hotkey.Registered) {
            _tray.BalloonTipTitle = "ReFind shortcut unavailable";
            _tray.BalloonTipText = "Alt+Space is already assigned. Double-click this tray icon to search.";
            _tray.ShowBalloonTip(5000);
        }

        _healthTimer = new System.Windows.Forms.Timer { Interval = 30000 };
        _healthTimer.Tick += async (_, _) => await _backend.EnsureRunningAsync();
        _healthTimer.Start();
        _ = _backend.EnsureRunningAsync();
    }

    protected override void ExitThreadCore()
    {
        _healthTimer.Stop();
        _healthTimer.Dispose();
        _tray.Visible = false;
        _tray.Dispose();
        _hotkey.Dispose();
        _overlay.Dispose();
        _backend.Dispose();
        base.ExitThreadCore();
    }

    private static async Task OpenBrowserAsync(string root)
    {
        if (!await BackendHost.IsRespondingAsync("http://127.0.0.1:3000/")) {
            var frontend = Path.Combine(root, "frontend");
            var node = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "nodejs", "node.exe");
            var next = Path.Combine(frontend, "node_modules", "next", "dist", "bin", "next");
            if (File.Exists(node) && File.Exists(next)) {
                Process.Start(new ProcessStartInfo(node) {
                    Arguments = $"\"{next}\" start --hostname 127.0.0.1 --port 3000",
                    WorkingDirectory = frontend,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden,
                });
            }
            for (var attempt = 0; attempt < 20 && !await BackendHost.IsRespondingAsync("http://127.0.0.1:3000/"); attempt++)
                await Task.Delay(500);
        }
        Process.Start(new ProcessStartInfo("http://127.0.0.1:3000/") { UseShellExecute = true });
    }

    private static string FindProjectRoot()
    {
        for (var folder = new DirectoryInfo(AppContext.BaseDirectory); folder != null; folder = folder.Parent) {
            if (Directory.Exists(Path.Combine(folder.FullName, "backend")) &&
                Directory.Exists(Path.Combine(folder.FullName, "frontend"))) return folder.FullName;
        }
        throw new DirectoryNotFoundException("ReFind project folder was not found.");
    }

    private static Icon CreateTrayIcon()
    {
        using var bitmap = new Bitmap(32, 32);
        using var graphics = Graphics.FromImage(bitmap);
        graphics.SmoothingMode = SmoothingMode.AntiAlias;
        using var background = new SolidBrush(Color.FromArgb(92, 78, 190));
        graphics.FillEllipse(background, 1, 1, 30, 30);
        using var font = new Font("Segoe UI", 18, FontStyle.Bold, GraphicsUnit.Pixel);
        TextRenderer.DrawText(graphics, "R", font, new Rectangle(6, 3, 23, 25), Color.White,
            TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPadding);
        var handle = bitmap.GetHicon();
        try { return (Icon)Icon.FromHandle(handle).Clone(); }
        finally { DestroyIcon(handle); }
    }

    [DllImport("user32.dll")]
    private static extern bool DestroyIcon(IntPtr handle);
}

internal sealed class HotkeyWindow : NativeWindow, IDisposable
{
    private const int WmHotkey = 0x0312;
    private const uint ModAlt = 0x0001;
    private const uint ModNoRepeat = 0x4000;
    private const uint VkSpace = 0x20;
    private readonly Action _activate;
    public bool Registered { get; }

    public HotkeyWindow(Action activate)
    {
        _activate = activate;
        CreateHandle(new CreateParams { Caption = "ReFind Hotkey" });
        Registered = RegisterHotKey(Handle, 1, ModAlt | ModNoRepeat, VkSpace);
    }

    protected override void WndProc(ref Message message)
    {
        if (message.Msg == WmHotkey && message.WParam == (IntPtr)1) {
            _activate();
            return;
        }
        base.WndProc(ref message);
    }

    public void Dispose()
    {
        if (Registered) UnregisterHotKey(Handle, 1);
        DestroyHandle();
    }

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool RegisterHotKey(IntPtr window, int id, uint modifiers, uint virtualKey);
    [DllImport("user32.dll")]
    private static extern bool UnregisterHotKey(IntPtr window, int id);
}

internal sealed class BackendHost : IDisposable
{
    private readonly string _backendDirectory;
    private Process? _ownedProcess;
    private bool _starting;
    public HttpClient Client { get; } = new() { BaseAddress = new Uri("http://127.0.0.1:8000"), Timeout = TimeSpan.FromSeconds(20) };

    public BackendHost(string root) => _backendDirectory = Path.Combine(root, "backend");

    public async Task EnsureRunningAsync()
    {
        if (_starting || (_ownedProcess is { HasExited: false })) return;
        _starting = true;
        try {
            if (await IsRespondingAsync("http://127.0.0.1:8000/api/health")) return;
            var python = Path.Combine(_backendDirectory, ".venv", "Scripts", "pythonw.exe");
            if (!File.Exists(python)) return;
            var start = new ProcessStartInfo(python) {
                Arguments = "-m uvicorn app.main:app --host 127.0.0.1 --port 8000",
                WorkingDirectory = _backendDirectory,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden,
            };
            start.Environment["LOCAL_FAST_MODE"] = "true";
            start.Environment["HF_HUB_OFFLINE"] = "1";
            start.Environment["TRANSFORMERS_OFFLINE"] = "1";
            _ownedProcess = Process.Start(start);
        } catch { /* Search displays a clear connection error if startup fails. */ }
        finally { _starting = false; }
    }

    public static async Task<bool> IsRespondingAsync(string url)
    {
        try {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
            using var response = await client.GetAsync(url);
            return response.IsSuccessStatusCode;
        } catch { return false; }
    }

    public void Dispose()
    {
        Client.Dispose();
        if (_ownedProcess is { HasExited: false }) _ownedProcess.Kill();
        _ownedProcess?.Dispose();
    }
}

internal sealed class SearchResponse
{
    [JsonPropertyName("results")]
    public List<SearchResult> Results { get; set; } = [];
}

internal sealed class SearchResult
{
    [JsonPropertyName("document_id")]
    public string DocumentId { get; set; } = "";
    [JsonPropertyName("filename")]
    public string Filename { get; set; } = "";
    [JsonPropertyName("filepath")]
    public string Filepath { get; set; } = "";
    [JsonPropertyName("file_type")]
    public string FileType { get; set; } = "";
    [JsonPropertyName("snippet")]
    public string Snippet { get; set; } = "";
}

internal sealed class SearchOverlay : Form
{
    private static readonly Color Transparent = Color.FromArgb(1, 2, 3);
    private static readonly Color Surface = Color.FromArgb(18, 19, 26);
    private static readonly Color Muted = Color.FromArgb(145, 146, 160);
    private readonly BackendHost _backend;
    private readonly Panel _card;
    private readonly Panel _items;
    private readonly TextBox _input;
    private readonly Label _hint;
    private readonly Label _footer;
    private readonly System.Windows.Forms.Timer _debounce;
    private CancellationTokenSource? _searchCancel;
    private List<SearchResult> _results = [];
    private int _selected;
    private bool _opening;

    public SearchOverlay(BackendHost backend)
    {
        _backend = backend;
        FormBorderStyle = FormBorderStyle.None;
        ShowInTaskbar = false;
        StartPosition = FormStartPosition.Manual;
        TopMost = true;
        BackColor = Transparent;
        TransparencyKey = Transparent;
        Width = 730;
        Height = 180;
        KeyPreview = true;
        AutoScaleMode = AutoScaleMode.None;

        _card = new Panel { Left = 8, Top = 8, Width = 714, Height = 164, BackColor = Surface };
        Controls.Add(_card);

        var glyph = new Label {
            Text = "⌕", Left = 21, Top = 15, Width = 34, Height = 42,
            Font = new Font("Segoe UI Symbol", 27), ForeColor = Color.FromArgb(170, 159, 255),
            BackColor = Surface, TextAlign = ContentAlignment.MiddleCenter,
        };
        _card.Controls.Add(glyph);

        _input = new TextBox {
            Left = 62, Top = 24, Width = 605, Height = 34,
            Font = new Font("Segoe UI", 14), ForeColor = Color.White,
            BackColor = Surface, BorderStyle = BorderStyle.None,
            PlaceholderText = "Search files and memories...",
        };
        _input.TextChanged += (_, _) => OnQueryChanged();
        _input.KeyDown += InputKeyDown;
        _card.Controls.Add(_input);

        var divider = new Panel { Left = 0, Top = 70, Width = 714, Height = 1, BackColor = Color.FromArgb(48, 48, 60) };
        _card.Controls.Add(divider);

        _items = new Panel { Left = 10, Top = 82, Width = 694, Height = 44, BackColor = Surface, AutoScroll = true };
        _card.Controls.Add(_items);
        _hint = new Label {
            Text = "Type a filename or describe what you remember", Left = 0, Top = 0,
            Width = 690, Height = 43, TextAlign = ContentAlignment.MiddleCenter,
            Font = new Font("Segoe UI", 10), ForeColor = Muted,
        };
        _items.Controls.Add(_hint);

        _footer = new Label {
            Text = "ReFind                                       ↑ ↓  Select     ↵  Open     Esc  Close",
            Left = 19, Top = 132, Width = 674, Height = 24,
            Font = new Font("Segoe UI", 9), ForeColor = Muted,
            BackColor = Surface, TextAlign = ContentAlignment.MiddleLeft,
        };
        _card.Controls.Add(_footer);

        _debounce = new System.Windows.Forms.Timer { Interval = 260 };
        _debounce.Tick += async (_, _) => { _debounce.Stop(); await SearchAsync(_input.Text.Trim()); };
        Deactivate += (_, _) => HideOverlay();
        ResizeCard(0);
    }

    public void ShowOverlay()
    {
        var screen = Screen.FromPoint(Cursor.Position).WorkingArea;
        Left = screen.Left + (screen.Width - Width) / 2;
        Top = screen.Top + Math.Min(135, (int)(screen.Height * 0.14));
        Show();
        Activate();
        _input.Focus();
        _input.SelectAll();
        if (_input.Text.Length > 0) OnQueryChanged();
    }

    public void HideOverlay()
    {
        _debounce.Stop();
        _searchCancel?.Cancel();
        Hide();
    }

    private void OnQueryChanged()
    {
        _debounce.Stop();
        _searchCancel?.Cancel();
        _results = [];
        _selected = 0;
        RenderResults();
        if (string.IsNullOrWhiteSpace(_input.Text)) SetHint("Type a filename or describe what you remember");
        else {
            SetHint("Searching your index...");
            _debounce.Start();
        }
    }

    private async Task SearchAsync(string query)
    {
        if (query.Length == 0) return;
        _searchCancel?.Dispose();
        _searchCancel = new CancellationTokenSource();
        var token = _searchCancel.Token;
        try {
            await _backend.EnsureRunningAsync();
            var ready = false;
            for (var attempt = 0; attempt < 30 && !token.IsCancellationRequested; attempt++) {
                if (await BackendHost.IsRespondingAsync("http://127.0.0.1:8000/api/health")) { ready = true; break; }
                SetHint("Starting EmbeddingGemma search engine...");
                await Task.Delay(1000, token);
            }
            if (!ready) throw new HttpRequestException("Search engine is not running.");
            using var response = await _backend.Client.PostAsJsonAsync("/api/search", new { query, top_k = 8 }, token);
            response.EnsureSuccessStatusCode();
            var data = await response.Content.ReadFromJsonAsync<SearchResponse>(cancellationToken: token);
            if (token.IsCancellationRequested || IsDisposed || !Visible || _input.Text.Trim() != query) return;
            _results = data?.Results ?? [];
            _selected = 0;
            RenderResults();
            if (_results.Count == 0) SetHint("No matching files");
        } catch (OperationCanceledException) { }
        catch (Exception error) {
            if (!token.IsCancellationRequested && !IsDisposed) SetHint($"Search unavailable: {error.Message}");
        }
    }

    private void OpenSelected()
    {
        if (_opening || _results.Count == 0) return;
        _opening = true;
        var selected = _results[_selected];
        try {
            if (!File.Exists(selected.Filepath)) throw new FileNotFoundException("Original file is unavailable.");
            Process.Start(new ProcessStartInfo(selected.Filepath) { UseShellExecute = true });
            HideOverlay();
        } catch (Exception error) {
            _results = [];
            RenderResults();
            SetHint($"Could not open file: {error.Message}");
        } finally { _opening = false; }
    }

    private void InputKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.KeyCode == Keys.Escape) { HideOverlay(); e.SuppressKeyPress = true; }
        else if (e.KeyCode == Keys.Down && _results.Count > 0) {
            _selected = (_selected + 1) % _results.Count;
            RenderSelection(); e.SuppressKeyPress = true;
        } else if (e.KeyCode == Keys.Up && _results.Count > 0) {
            _selected = (_selected - 1 + _results.Count) % _results.Count;
            RenderSelection(); e.SuppressKeyPress = true;
        } else if (e.KeyCode == Keys.Enter) { OpenSelected(); e.SuppressKeyPress = true; }
    }

    private void RenderResults()
    {
        _items.SuspendLayout();
        foreach (Control control in _items.Controls.Cast<Control>().Where(control => control != _hint).ToArray()) {
            _items.Controls.Remove(control);
            control.Dispose();
        }
        _items.AutoScrollPosition = Point.Empty;
        _hint.Visible = _results.Count == 0;
        for (var index = 0; index < _results.Count; index++) {
            var rowIndex = index;
            var row = new ResultRow(_results[index]) {
                Left = 1, Top = index * 62, Width = 672, Height = 59, Selected = index == _selected,
            };
            row.MouseEnter += (_, _) => { _selected = rowIndex; RenderSelection(); };
            row.Click += (_, _) => OpenSelected();
            _items.Controls.Add(row);
        }
        ResizeCard(_results.Count);
        _items.ResumeLayout();
    }

    private void RenderSelection()
    {
        foreach (var row in _items.Controls.OfType<ResultRow>().Select((item, index) => (item, index))) {
            row.item.Selected = row.index == _selected;
            row.item.Invalidate();
            if (row.index == _selected) _items.ScrollControlIntoView(row.item);
        }
    }

    private void SetHint(string message)
    {
        _hint.Text = message;
        _hint.Visible = true;
    }

    private void ResizeCard(int count)
    {
        var resultHeight = count == 0 ? 44 : Math.Min(count * 62, 496);
        _items.Height = resultHeight;
        _card.Height = 82 + resultHeight + 39;
        _footer.Top = _card.Height - 31;
        Height = _card.Height + 16;
        using var path = RoundedRectangle(new Rectangle(0, 0, _card.Width, _card.Height), 16);
        _card.Region?.Dispose();
        _card.Region = new Region(path);
    }

    private static GraphicsPath RoundedRectangle(Rectangle rect, int radius)
    {
        var path = new GraphicsPath();
        var d = radius * 2;
        path.AddArc(rect.X, rect.Y, d, d, 180, 90);
        path.AddArc(rect.Right - d, rect.Y, d, d, 270, 90);
        path.AddArc(rect.Right - d, rect.Bottom - d, d, d, 0, 90);
        path.AddArc(rect.X, rect.Bottom - d, d, d, 90, 90);
        path.CloseFigure();
        return path;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) { _debounce.Dispose(); _searchCancel?.Dispose(); }
        base.Dispose(disposing);
    }
}

internal sealed class ResultRow(SearchResult result) : Control
{
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public bool Selected { get; set; }

    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        using var background = new SolidBrush(Selected ? Color.FromArgb(38, 35, 65) : Color.FromArgb(18, 19, 26));
        using var path = new GraphicsPath();
        path.AddArc(0, 0, 16, 16, 180, 90);
        path.AddArc(Width - 16, 0, 16, 16, 270, 90);
        path.AddArc(Width - 16, Height - 16, 16, 16, 0, 90);
        path.AddArc(0, Height - 16, 16, 16, 90, 90);
        path.CloseFigure();
        e.Graphics.FillPath(background, path);

        using var iconBrush = new SolidBrush(Color.FromArgb(44, 43, 61));
        e.Graphics.FillRectangle(iconBrush, 11, 11, 37, 37);
        var extension = result.FileType.ToUpperInvariant();
        using var iconFont = new Font("Segoe UI", 8, FontStyle.Bold);
        TextRenderer.DrawText(e.Graphics, extension.Length > 4 ? extension[..4] : extension, iconFont,
            new Rectangle(13, 14, 33, 30), Color.FromArgb(186, 176, 255), TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter);

        using var titleFont = new Font("Segoe UI", 10, FontStyle.Bold);
        using var detailFont = new Font("Segoe UI", 8.5f);
        TextRenderer.DrawText(e.Graphics, result.Filename, titleFont, new Rectangle(60, 8, Width - 145, 23),
            Color.White, TextFormatFlags.EndEllipsis | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPrefix);
        TextRenderer.DrawText(e.Graphics, string.IsNullOrWhiteSpace(result.Snippet) ? result.Filepath : result.Snippet,
            detailFont, new Rectangle(60, 31, Width - 110, 19), Color.FromArgb(151, 152, 164),
            TextFormatFlags.EndEllipsis | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPrefix);
        TextRenderer.DrawText(e.Graphics, extension, detailFont, new Rectangle(Width - 80, 17, 67, 22),
            Color.FromArgb(151, 152, 164), TextFormatFlags.Right | TextFormatFlags.VerticalCenter);
    }
}
