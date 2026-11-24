"""FundoZero: editor de remoção local de fundo por IA para Windows."""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from tkinter import (
    BOTH, BOTTOM, CENTER, END, HORIZONTAL, LEFT, RIGHT, TOP, X, Y,
    BooleanVar, Canvas, Frame, IntVar, Label, StringVar, Tk,
    filedialog, messagebox, ttk,
)

from PIL import Image, ImageTk
from motor import EXTENSOES, MODELOS, Traco, abrir_imagem, finalizar_imagem, remover_fundo
from lote import gerar_lote
from midia import EXTENSOES_SUPORTADAS, tipo_arquivo, primeira_imagem, processar_midia

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    JanelaBase = TkinterDnD.Tk
    DRAG_DROP = True
except ImportError:
    JanelaBase = Tk
    DRAG_DROP = False

FUNDO = '#10141f'
PAINEL = '#1a2130'
CARTAO = '#212b3b'
TEXTO = '#f0f4fa'
SECUNDARIO = '#a4b1c5'
DESTAQUE = '#5cb9f8'
VERDE = '#47d6b0'
BORDA = '#344256'


def erro_no_log(texto: str):
    try:
        with (Path(__file__).resolve().parent / 'erros.log').open('a', encoding='utf-8') as arquivo:
            arquivo.write(texto + '\n')
    except OSError:
        pass


def botao(pai, texto, comando, cor=CARTAO, frente=TEXTO, padx=12, pady=11):
    import tkinter as tk
    btn = tk.Button(
        pai, text=texto, command=comando, bg=cor, fg=frente, activebackground=DESTAQUE,
        activeforeground='#09101e', font=('Segoe UI', 10, 'bold'), relief='flat', bd=0,
        cursor='hand2', padx=padx, pady=pady, highlightthickness=0, disabledforeground='#8993a5'
    )
    return btn


class FundoZero(JanelaBase):
    def __init__(self):
        super().__init__()
        self.title('FundoZero • Remoção local de background')
        self.geometry('1340x820')
        self.minsize(1080, 680)
        self.configure(bg=FUNDO)
        self._configurar_estilo()

        self.mensagens = queue.Queue()
        self.sessoes = {}  # Acesso apenas na thread de trabalho
        self.ocupado = False
        self.caminho_atual: Path | None = None
        self.tipo_atual = 'imagem'
        self.original: Image.Image | None = None
        self.recorte: Image.Image | None = None
        self.final: Image.Image | None = None
        self.tracos: list[Traco] = []
        self.traco_em_andamento: list[tuple[int, int]] | None = None
        self._image_before = None
        self._image_after = None
        self._pos_image = (0, 0, 1)  # x0, y0, escala
        self._preview_later = None
        self._render_later = None
        self._canvas_size = (1, 1)

        self.modelo = StringVar(value='Equilibrado | Fotos e objetos')
        self.modo_lote = StringVar(value='zip')
        self.formato_gif = StringVar(value='gif')
        self.descontaminar = BooleanVar(value=True)
        self.borda = IntVar(value=0)
        self.suavidade = IntVar(value=0)
        self.limpeza = IntVar(value=0)
        self.tamanho_pincel = IntVar(value=38)
        self.tipo_pincel = StringVar(value='navegar')
        self.fundo_preview = StringVar(value='Xadrez')
        self.mensagem_status = StringVar(value='Importe uma imagem, GIF ou vídeo para começar.')
        self.arquivo_texto = StringVar(value='Nenhum arquivo selecionado')
        self.info_imagem = StringVar(value='PNG • JPG • WEBP • GIF animado • MP4 • MOV • WEBM • MKV')

        self._criar_interface()
        self.after(100, self._ler_mensagens)
        self.protocol('WM_DELETE_WINDOW', self._fechar)

    def _configurar_estilo(self):
        estilo = ttk.Style(self)
        if 'clam' in estilo.theme_names():
            estilo.theme_use('clam')
        estilo.configure('FundoZero.TCombobox', fieldbackground=CARTAO, background=CARTAO,
                         foreground=TEXTO, arrowcolor=TEXTO, bordercolor=BORDA, padding=7)
        estilo.map('FundoZero.TCombobox', fieldbackground=[('readonly', CARTAO)],
                   foreground=[('readonly', TEXTO)])
        estilo.configure('FundoZero.Horizontal.TProgressbar', troughcolor=FUNDO,
                         background=VERDE, lightcolor=VERDE, darkcolor=VERDE, bordercolor=FUNDO)

    def _criar_interface(self):
        cabecalho = Frame(self, bg=FUNDO, height=80)
        cabecalho.pack(side=TOP, fill=X, padx=24, pady=(14, 9))
        cabecalho.pack_propagate(False)
        bloco = Frame(cabecalho, bg=FUNDO)
        bloco.pack(side=LEFT, anchor='w')
        Label(bloco, text='FUNDO', bg=FUNDO, fg=TEXTO, font=('Segoe UI', 23, 'bold')).pack(side=LEFT)
        Label(bloco, text='ZERO', bg=FUNDO, fg=DESTAQUE, font=('Segoe UI', 23, 'bold')).pack(side=LEFT)
        Label(cabecalho, text='ESTÚDIO LOCAL  •  SEM API  •  PNG / GIF / WEBP / WEBM',
              fg=SECUNDARIO, bg=FUNDO, font=('Segoe UI', 9, 'bold')).pack(side=RIGHT, anchor='e', pady=15)

        corpo = Frame(self, bg=FUNDO)
        corpo.pack(fill=BOTH, expand=True, padx=18, pady=(0, 10))
        lateral = Frame(corpo, bg=PAINEL, width=335)
        lateral.pack(side=LEFT, fill=Y, padx=(0, 13))
        lateral.pack_propagate(False)
        self._painel_controles(lateral)

        principal = Frame(corpo, bg=FUNDO)
        principal.pack(side=LEFT, fill=BOTH, expand=True)
        titulo = Frame(principal, bg=FUNDO)
        titulo.pack(fill=X, pady=(0, 9))
        Label(titulo, text='PRÉVIA', fg=TEXTO, bg=FUNDO, font=('Segoe UI', 11, 'bold')).pack(side=LEFT)
        opcoes = ttk.Combobox(titulo, textvariable=self.fundo_preview,
                              values=['Xadrez', 'Branco', 'Escuro'], state='readonly',
                              style='FundoZero.TCombobox', width=10)
        opcoes.pack(side=RIGHT)
        opcoes.bind('<<ComboboxSelected>>', lambda _e: self._agendar_preview())
        Label(titulo, text='Fundo da prévia', fg=SECUNDARIO, bg=FUNDO,
              font=('Segoe UI', 9)).pack(side=RIGHT, padx=(0, 9))

        quadros = Frame(principal, bg=FUNDO)
        quadros.pack(fill=BOTH, expand=True)
        original_painel = Frame(quadros, bg=PAINEL, highlightbackground=BORDA, highlightthickness=1)
        original_painel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 6))
        Label(original_painel, text='ORIGINAL', bg=PAINEL, fg=SECUNDARIO,
              font=('Segoe UI', 9, 'bold')).pack(fill=X, padx=16, pady=(12, 7))
        self.canvas_antes = Canvas(original_painel, bg='#273243', highlightthickness=0, bd=0)
        self.canvas_antes.pack(fill=BOTH, expand=True, padx=11, pady=(0, 11))
        self.canvas_antes.bind('<Configure>', lambda _e: self._agendar_preview())

        resultado_painel = Frame(quadros, bg=PAINEL, highlightbackground=DESTAQUE, highlightthickness=1)
        resultado_painel.pack(side=LEFT, fill=BOTH, expand=True, padx=(6, 0))
        Label(resultado_painel, text='TRANSPARENTE  •  PINCEL EDITÁVEL', bg=PAINEL, fg=DESTAQUE,
              font=('Segoe UI', 9, 'bold')).pack(fill=X, padx=16, pady=(12, 7))
        self.canvas_depois = Canvas(resultado_painel, bg='#273243', highlightthickness=0, bd=0,
                                   cursor='crosshair')
        self.canvas_depois.pack(fill=BOTH, expand=True, padx=11, pady=(0, 11))
        self.canvas_depois.bind('<Configure>', lambda _e: self._agendar_preview())
        self.canvas_depois.bind('<ButtonPress-1>', self._inicio_pincel)
        self.canvas_depois.bind('<B1-Motion>', self._movimento_pincel)
        self.canvas_depois.bind('<ButtonRelease-1>', self._fim_pincel)

        status_frame = Frame(principal, bg=PAINEL, height=61)
        status_frame.pack(fill=X, pady=(11, 0))
        status_frame.pack_propagate(False)
        Label(status_frame, textvariable=self.mensagem_status, bg=PAINEL, fg=TEXTO,
              font=('Segoe UI', 10), anchor='w').pack(side=TOP, fill=X, padx=13, pady=(8, 3))
        self.progresso = ttk.Progressbar(status_frame, mode='indeterminate', maximum=100,
                                        style='FundoZero.Horizontal.TProgressbar')
        self.progresso.pack(side=BOTTOM, fill=X, padx=13, pady=(0, 8))
        self.progresso['value'] = 0

        if DRAG_DROP:
            for area in (self.canvas_antes, self.canvas_depois):
                area.drop_target_register(DND_FILES)
                area.dnd_bind('<<Drop>>', self._arrastar_arquivo)
        self._agendar_preview()

    def _secao(self, pai, nome):
        Label(pai, text=nome, fg=DESTAQUE, bg=PAINEL,
              font=('Segoe UI', 9, 'bold'), anchor='w').pack(fill=X, pady=(16, 8))

    def _painel_controles(self, pai):
        import tkinter as tk
        rolador = Canvas(pai, bg=PAINEL, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(pai, orient='vertical', command=rolador.yview)
        interior = Frame(rolador, bg=PAINEL)
        janela = rolador.create_window((0, 0), window=interior, anchor='nw')
        rolador.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        rolador.pack(side=LEFT, fill=BOTH, expand=True)
        interior.bind('<Configure>', lambda _e: rolador.configure(scrollregion=rolador.bbox('all')))
        rolador.bind('<Configure>', lambda e: rolador.itemconfig(janela, width=e.width))
        def rolar(event):
            rolador.yview_scroll(-1 if event.delta > 0 else 1, 'units')
        def ligar_rolagem(widget):
            widget.bind('<MouseWheel>', rolar)
            for child in widget.winfo_children():
                if not isinstance(child, ttk.Combobox):
                    ligar_rolagem(child)
        self.after(1000, lambda: ligar_rolagem(interior))
        area = Frame(interior, bg=PAINEL)
        area.pack(fill=X, padx=16, pady=10)

        self._secao(area, '01  •  ENTRADA')
        self.btn_abrir = botao(area, '＋  Importar imagem, GIF ou vídeo', self._selecionar_imagem, cor=DESTAQUE, frente='#0c1725')
        self.btn_abrir.pack(fill=X)
        Label(area, textvariable=self.arquivo_texto, bg=PAINEL, fg=TEXTO,
              wraplength=265, justify=LEFT, font=('Segoe UI', 9)).pack(fill=X, pady=(8, 2))
        Label(area, textvariable=self.info_imagem, bg=PAINEL, fg=SECUNDARIO,
              wraplength=270, justify=LEFT, font=('Segoe UI', 8)).pack(fill=X)

        self._secao(area, '02  •  INTELIGÊNCIA ARTIFICIAL')
        self.combo_modelo = ttk.Combobox(area, textvariable=self.modelo,
                                         values=list(MODELOS), state='readonly',
                                         style='FundoZero.TCombobox')
        self.combo_modelo.pack(fill=X)
        self.chk_halo = tk.Checkbutton(area, text='Corrigir halo de cor nas bordas', variable=self.descontaminar,
                                      bg=PAINEL, fg=TEXTO, selectcolor=CARTAO,
                                      activebackground=PAINEL, activeforeground=TEXTO,
                                      highlightthickness=0, font=('Segoe UI', 9))
        self.chk_halo.pack(anchor='w', pady=(9, 8))
        self.btn_remover = botao(area, '✦  Remover fundo com IA', self._iniciar_remocao,
                                 cor=VERDE, frente='#072019')
        self.btn_remover.pack(fill=X)
        Label(area, text='Em GIFs e vídeos, este botão faz a prévia de um quadro.', bg=PAINEL, fg=SECUNDARIO,
              font=('Segoe UI', 8)).pack(anchor='w', pady=(6, 0))

        self._secao(area, '03  •  AJUSTES FINOS')
        self._controle(area, 'Ajustar borda (− contrai / + expande)', self.borda, -6, 6)
        self._controle(area, 'Suavidade do contorno', self.suavidade, 0, 6)
        self._controle(area, 'Limpar resíduos transparentes', self.limpeza, 0, 90)

        self._secao(area, '04  •  CORREÇÃO MANUAL')
        ferramentas = Frame(area, bg=PAINEL)
        ferramentas.pack(fill=X)
        self.btn_apagar = botao(ferramentas, 'Apagar', lambda: self._selecionar_pincel('apagar'), padx=7, pady=7)
        self.btn_apagar.pack(side=LEFT, fill=X, expand=True, padx=(0, 4))
        self.btn_restaurar = botao(ferramentas, 'Restaurar', lambda: self._selecionar_pincel('restaurar'), padx=7, pady=7)
        self.btn_restaurar.pack(side=LEFT, fill=X, expand=True, padx=4)
        self.btn_navegar = botao(ferramentas, 'Nenhum', lambda: self._selecionar_pincel('navegar'), padx=7, pady=7)
        self.btn_navegar.pack(side=LEFT, fill=X, expand=True, padx=(4, 0))
        self._atualizar_botoes_pincel()
        self._controle(area, 'Tamanho do pincel (pixels)', self.tamanho_pincel, 4, 180, atualizar=False)
        desfazer = Frame(area, bg=PAINEL)
        desfazer.pack(fill=X, pady=(3, 0))
        self.btn_desfazer = botao(desfazer, '↶  Desfazer', self._desfazer, padx=7, pady=7)
        self.btn_desfazer.pack(side=LEFT, fill=X, expand=True, padx=(0, 4))
        self.btn_limpar = botao(desfazer, 'Limpar pincéis', self._limpar_pinceis, padx=7, pady=7)
        self.btn_limpar.pack(side=LEFT, fill=X, expand=True, padx=(4, 0))

        self._secao(area, '05  •  EXPORTAÇÃO')
        self.btn_exportar = botao(area, '⬇  Exportar PNG transparente', self._exportar, cor=DESTAQUE,
                                  frente='#0c1725')
        self.btn_exportar.pack(fill=X)
        self.btn_animacao = botao(area, '▶  Processar GIF / vídeo completo', self._exportar_animacao,
                                  cor=VERDE, frente='#072019')
        self.btn_animacao.pack(fill=X, pady=(7, 6))
        self.btn_animacao.configure(state='disabled')
        Label(area, text='Saída de GIF: GIF (alfa simples) ou WebP (alfa suave)', fg=SECUNDARIO,
              bg=PAINEL, wraplength=274, font=('Segoe UI', 8)).pack(anchor='w', pady=(5, 3))
        self.combo_formato_gif = ttk.Combobox(area, textvariable=self.formato_gif,
                values=['gif', 'webp'], state='readonly', style='FundoZero.TCombobox')
        self.combo_formato_gif.pack(fill=X)
        Label(area, text='Pincéis na prévia só afetam o PNG estático; na animação valem os ajustes de borda.',
              fg=SECUNDARIO, bg=PAINEL, wraplength=274,
              font=('Segoe UI', 8)).pack(anchor='w', pady=(4, 3))
        Label(area, text='Saída de vídeos: WebM VP9 com transparência e áudio (quando existir).',
              fg=SECUNDARIO, bg=PAINEL, wraplength=274,
              font=('Segoe UI', 8)).pack(anchor='w', pady=(6, 3))
        Label(area, text='Ao processar vários arquivos, salvar como:', fg=SECUNDARIO,
              bg=PAINEL, font=('Segoe UI', 9)).pack(anchor='w', pady=(11, 3))
        for texto, valor in (('Um ZIP com PNGs, GIFs/WebPs e vídeos', 'zip'),
                             ('Arquivos separados em uma pasta', 'pasta')):
            tk.Radiobutton(area, text=texto, value=valor, variable=self.modo_lote,
                           bg=PAINEL, fg=TEXTO, selectcolor=CARTAO,
                           activebackground=PAINEL, activeforeground=TEXTO,
                           highlightthickness=0, font=('Segoe UI', 9)).pack(anchor='w')
        self.btn_lote = botao(area, '▣  Remover fundos em lote', self._processar_pasta)
        self.btn_lote.pack(fill=X, pady=(7, 4))
        Label(area, text='Tudo é processado localmente. A primeira instalação requer internet. Vídeos podem demorar: o recorte é quadro a quadro.',
              wraplength=270, justify=LEFT, bg=PAINEL, fg=SECUNDARIO,
              font=('Segoe UI', 8)).pack(fill=X, pady=(9, 16))

    def _controle(self, pai, legenda, variavel, minimo, maximo, atualizar=True):
        import tkinter as tk
        linha = Frame(pai, bg=PAINEL)
        linha.pack(fill=X, pady=(5, 1))
        Label(linha, text=legenda, fg=TEXTO, bg=PAINEL, font=('Segoe UI', 8),
              anchor='w').pack(side=LEFT)
        Label(linha, textvariable=variavel, fg=DESTAQUE, bg=PAINEL,
              font=('Segoe UI', 9, 'bold'), width=4).pack(side=RIGHT)
        escala = tk.Scale(pai, from_=minimo, to=maximo, orient=HORIZONTAL, variable=variavel,
                          command=(lambda _v: self._agendar_preview()) if atualizar else None,
                          showvalue=False, bg=PAINEL, fg=TEXTO, activebackground=DESTAQUE,
                          highlightthickness=0, troughcolor=BORDA, bd=0, sliderlength=15,
                          resolution=1)
        escala.pack(fill=X, pady=(0, 4))

    def _arrastar_arquivo(self, event):
        try:
            caminhos = self.tk.splitlist(event.data)
            for caminho in caminhos:
                arquivo = Path(caminho)
                if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_SUPORTADAS:
                    self._carregar_arquivo(arquivo)
                    break
        except Exception as ex:
            self.mensagem_status.set(f'Erro no arrastar e soltar: {ex}')

    def _selecionar_imagem(self):
        if self.ocupado:
            return
        arquivo = filedialog.askopenfilename(title='Escolha imagem, GIF ou vídeo',
            filetypes=[('Imagens, GIFs e vídeos', '*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif *.mp4 *.mov *.mkv *.webm *.avi *.m4v *.wmv *.mpeg *.mpg'),
                       ('Imagens estáticas', '*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff'),
                       ('Animações', '*.gif *.webp'),
                       ('Vídeos', '*.mp4 *.mov *.mkv *.webm *.avi *.m4v *.wmv *.mpeg *.mpg'),
                       ('Todos os arquivos', '*.*')])
        if arquivo:
            self._carregar_arquivo(Path(arquivo))

    def _carregar_arquivo(self, arquivo: Path):
        if self.ocupado:
            return
        try:
            tipo = tipo_arquivo(arquivo)
            original = primeira_imagem(arquivo)
            if original.width * original.height > 100_000_000:
                messagebox.showwarning('Imagem grande', 'Imagens acima de 100 megapixels podem consumir muita memória.')
            self.original = original
            self.caminho_atual = arquivo
            self.tipo_atual = tipo
            self.recorte = None
            self.final = None
            self.tracos.clear()
            self.arquivo_texto.set(arquivo.name)
            detalhe = f'{original.width:,} × {original.height:,} pixels'.replace(',', '.')
            if tipo != 'imagem':
                detalhe += ' • Prévia do 1º quadro • Exporte o arquivo completo com botão verde'
                self.btn_remover.configure(text='✦  Prévia: remover fundo do 1º quadro')
                self.btn_exportar.configure(text='⬇  Salvar 1º quadro como PNG')
                self.btn_animacao.configure(state='normal')
                self.mensagem_status.set('Arquivo animado carregado. Use “Processar GIF / vídeo completo” para todos os quadros.')
            else:
                self.btn_remover.configure(text='✦  Remover fundo com IA')
                self.btn_exportar.configure(text='⬇  Exportar PNG transparente')
                self.btn_animacao.configure(state='disabled')
                self.mensagem_status.set('Imagem carregada. Clique em “Remover fundo com IA”.')
            self.info_imagem.set(detalhe)
            self._agendar_preview()
        except Exception as ex:
            messagebox.showerror('Não foi possível abrir', str(ex))

    def _dados_ajustes(self):
        return dict(borda=self.borda.get(), suavidade=self.suavidade.get(), limpeza=self.limpeza.get())

    def _iniciar_remocao(self):
        if self.ocupado:
            return
        if self.original is None:
            messagebox.showinfo('Escolha uma imagem', 'Primeiro importe uma imagem.')
            return
        model = MODELOS[self.modelo.get()]
        self._marcar_ocupado(True, 'Carregando IA local. No primeiro uso, o modelo será baixado...')
        original = self.original.copy()
        threading.Thread(target=self._remover_thread,
                         args=(original, model, self.descontaminar.get()), daemon=True).start()

    def _obter_sessao(self, nome):
        if nome not in self.sessoes:
            from rembg import new_session
            self.sessoes[nome] = new_session(nome)
        return self.sessoes[nome]

    def _remover_thread(self, original, modelo, descontaminar):
        try:
            sessao = self._obter_sessao(modelo)
            self.mensagens.put(('status', 'Analisando contornos e separando o objeto...'))
            recorte = remover_fundo(original, sessao, descontaminar)
            self.mensagens.put(('concluido', recorte))
        except Exception:
            mensagem = traceback.format_exc()
            erro_no_log(mensagem)
            self.mensagens.put(('erro', mensagem))

    def _marcar_ocupado(self, estado, mensagem=''):
        self.ocupado = estado
        if mensagem:
            self.mensagem_status.set(mensagem)
        for button in (self.btn_abrir, self.btn_remover, self.btn_exportar, self.btn_lote, self.btn_animacao):
            button.configure(state='disabled' if estado else 'normal')
        if not estado:
            self.btn_animacao.configure(state='normal' if self.tipo_atual != 'imagem' else 'disabled')
        self.combo_modelo.configure(state='disabled' if estado else 'readonly')
        self.combo_formato_gif.configure(state='disabled' if estado else 'readonly')
        if estado:
            self.progresso.configure(mode='indeterminate')
            self.progresso.start(12)
        else:
            self.progresso.stop()
            self.progresso['value'] = 0

    def _ler_mensagens(self):
        try:
            while True:
                mensagem = self.mensagens.get_nowait()
                tipo = mensagem[0]
                if tipo == 'status':
                    self.mensagem_status.set(mensagem[1])
                elif tipo == 'concluido':
                    self.recorte = mensagem[1]
                    self.tracos.clear()
                    self._marcar_ocupado(False, 'Recorte concluído. Ajuste bordas ou use o pincel e exporte.')
                    self._agendar_preview()
                elif tipo == 'erro':
                    self._marcar_ocupado(False, 'Não foi possível processar. Consulte erros.log.')
                    mensagem_curta = mensagem[1].splitlines()[-1] if mensagem[1].splitlines() else 'Erro desconhecido'
                    messagebox.showerror('Falha na IA local', f'{mensagem_curta}\n\nSe foi o primeiro uso, confirme a conexão à internet e tente novamente. Detalhes em erros.log.')
                elif tipo == 'midia_concluida':
                    _, destino, quadros = mensagem
                    self._marcar_ocupado(False, f'Concluído: {quadros} quadro(s) exportado(s).')
                    messagebox.showinfo('Animação transparente pronta',
                        f'{quadros} quadro(s) processado(s).\nArquivo salvo em:\n{destino}')
                elif tipo == 'midia_erro':
                    self._marcar_ocupado(False, 'Falha ao processar a animação. Consulte erros.log.')
                    messagebox.showerror('Falha no processamento', mensagem[1][-1100:])
                elif tipo == 'lote_progress':
                    self.mensagem_status.set(mensagem[1])
                elif tipo == 'lote_concluido':
                    _, acertos, falhas, destino, em_zip = mensagem
                    self._marcar_ocupado(False, f'Lote finalizado: {acertos} arquivo(s); {len(falhas)} erro(s).')
                    detalhe = '\n'.join(falhas[:8]) or 'Sem erros.'
                    rotulo = 'Arquivo ZIP' if em_zip else 'Pasta de resultados'
                    info = f'{rotulo}: {destino}' if acertos else 'Nenhum arquivo de saída foi gerado.'
                    messagebox.showinfo('Processamento concluído',
                        f'{acertos} arquivo(s) processado(s).\n{info}\n\n{detalhe}')
        except queue.Empty:
            pass
        self.after(100, self._ler_mensagens)

    def _agendar_preview(self):
        if self._preview_later is not None:
            self.after_cancel(self._preview_later)
        self._preview_later = self.after(160, self._renderizar_preview)

    def _renderizar_preview(self):
        self._preview_later = None
        if self.original is None:
            for canvas, texto in ((self.canvas_antes, 'Importe sua imagem aqui'),
                                   (self.canvas_depois, 'O PNG transparente aparecerá aqui')):
                canvas.delete('all')
                w = max(canvas.winfo_width(), 220)
                h = max(canvas.winfo_height(), 220)
                canvas.create_text(w // 2, h // 2, text=texto, fill=SECUNDARIO,
                                   font=('Segoe UI', 12), width=w - 25)
            return
        try:
            self.final = finalizar_imagem(self.original, self.recorte, **self._dados_ajustes(),
                           tracos=self.tracos) if self.recorte is not None else None
            self._renderizar_canvas(self.canvas_antes, self.original, depois=False)
            self._renderizar_canvas(self.canvas_depois, self.final or self.original, depois=True)
        except Exception as exc:
            erro_no_log(traceback.format_exc())
            self.mensagem_status.set(f'Falha na prévia: {exc}')

    def _renderizar_canvas(self, canvas, img, depois):
        canvas.delete('all')
        w, h = max(canvas.winfo_width(), 10), max(canvas.winfo_height(), 10)
        escala = min((w - 32) / img.width, (h - 32) / img.height)
        escala = max(escala, 0.001)
        iw, ih = max(1, round(img.width * escala)), max(1, round(img.height * escala))
        x0, y0 = round((w - iw) / 2), round((h - ih) / 2)
        cor_fundo = self.fundo_preview.get() if depois else 'Escuro'
        if cor_fundo == 'Xadrez' and depois:
            tamanho = 22
            for y in range(y0, y0 + ih, tamanho):
                for x in range(x0, x0 + iw, tamanho):
                    ind = ((x - x0) // tamanho + (y - y0) // tamanho) % 2
                    canvas.create_rectangle(x, y, min(x + tamanho, x0 + iw),
                                            min(y + tamanho, y0 + ih), outline='',
                                            fill='#f0f2f4' if ind else '#cdd3db')
        else:
            cor = '#ffffff' if cor_fundo == 'Branco' else '#293445'
            canvas.create_rectangle(x0, y0, x0 + iw, y0 + ih, outline='', fill=cor)
        imagem_reduzida = img.resize((iw, ih), Image.Resampling.LANCZOS)
        foto = ImageTk.PhotoImage(imagem_reduzida)
        canvas.create_image(x0, y0, image=foto, anchor='nw')
        if depois:
            self._image_after = foto
            self._pos_image = (x0, y0, escala)
        else:
            self._image_before = foto
        canvas.create_rectangle(x0, y0, x0 + iw, y0 + ih, outline=BORDA, width=1)
        if depois and self.recorte is None:
            canvas.create_text(w // 2, h - 26, text='Aguardando a remoção automática...',
                               fill=TEXTO, font=('Segoe UI', 10, 'bold'))

    def _selecionar_pincel(self, modo):
        self.tipo_pincel.set(modo)
        self._atualizar_botoes_pincel()

    def _atualizar_botoes_pincel(self):
        for nome, btn in (('apagar', self.btn_apagar), ('restaurar', self.btn_restaurar), ('navegar', self.btn_navegar)):
            escolhido = nome == self.tipo_pincel.get()
            btn.configure(bg=DESTAQUE if escolhido else CARTAO,
                          fg='#0c1725' if escolhido else TEXTO)

    def _mapa_xy(self, event):
        if self.original is None:
            return None
        x0, y0, escala = self._pos_image
        px = round((event.x - x0) / escala)
        py = round((event.y - y0) / escala)
        if 0 <= px < self.original.width and 0 <= py < self.original.height:
            return (px, py)
        return None

    def _inicio_pincel(self, event):
        if self.ocupado or self.recorte is None or self.tipo_pincel.get() == 'navegar':
            return
        ponto = self._mapa_xy(event)
        if ponto is not None:
            self.traco_em_andamento = [ponto]
            self._mostrar_brush_overlay(event)

    def _movimento_pincel(self, event):
        if self.traco_em_andamento is None:
            return
        ponto = self._mapa_xy(event)
        if ponto is not None:
            self.traco_em_andamento.append(ponto)
            self._mostrar_brush_overlay(event)

    def _mostrar_brush_overlay(self, event):
        self.canvas_depois.delete('pincel_tmp')
        if self.traco_em_andamento is None:
            return
        x0, y0, escala = self._pos_image
        pts = [(x0 + x * escala, y0 + y * escala) for x, y in self.traco_em_andamento]
        if len(pts) > 1:
            self.canvas_depois.create_line(*(coordenada for ponto in pts for coordenada in ponto),
                width=max(2, int(self.tamanho_pincel.get() * escala)),
                fill='#ff6c75' if self.tipo_pincel.get() == 'apagar' else VERDE,
                capstyle='round', joinstyle='round', tags='pincel_tmp')
        raio = max(3, int(self.tamanho_pincel.get() * escala / 2))
        self.canvas_depois.create_oval(event.x - raio, event.y - raio, event.x + raio, event.y + raio,
             outline='#ff6c75' if self.tipo_pincel.get() == 'apagar' else VERDE,
             width=2, tags='pincel_tmp')

    def _fim_pincel(self, event):
        if self.traco_em_andamento is None:
            return
        ultimo = self._mapa_xy(event)
        if ultimo is not None:
            self.traco_em_andamento.append(ultimo)
        pontos = tuple(self.traco_em_andamento)
        self.traco_em_andamento = None
        self.canvas_depois.delete('pincel_tmp')
        if pontos:
            self.tracos.append(Traco(self.tipo_pincel.get(), self.tamanho_pincel.get(), pontos))
            self._agendar_preview()

    def _desfazer(self):
        if self.ocupado:
            return
        if self.tracos:
            self.tracos.pop()
            self._agendar_preview()

    def _limpar_pinceis(self):
        if self.ocupado:
            return
        self.tracos.clear()
        self._agendar_preview()

    def _exportar(self):
        if self.ocupado:
            return
        if self.recorte is None or self.caminho_atual is None:
            messagebox.showinfo('Remova o fundo primeiro', 'Execute a remoção com IA antes de exportar.')
            return
        sugestao = self.caminho_atual.stem + '_sem_fundo.png'
        destino = filedialog.asksaveasfilename(title='Exportar PNG com transparência',
                initialdir=str(self.caminho_atual.parent), initialfile=sugestao,
                defaultextension='.png', filetypes=[('PNG transparente', '*.png')])
        if not destino:
            return
        try:
            resultado = finalizar_imagem(self.original, self.recorte,
                         **self._dados_ajustes(), tracos=self.tracos)
            resultado.save(destino, format='PNG', optimize=True)
            self.mensagem_status.set(f'PNG transparente exportado: {Path(destino).name}')
            messagebox.showinfo('Arquivo salvo', f'Imagem exportada com transparência real (canal alfa).\n\n{destino}')
        except Exception as exc:
            erro_no_log(traceback.format_exc())
            messagebox.showerror('Falha ao salvar', str(exc))

    def _exportar_animacao(self):
        if self.ocupado or self.caminho_atual is None:
            return
        if self.tipo_atual not in ('gif', 'video'):
            messagebox.showinfo('Arquivo estático', 'Para imagens estáticas, use a exportação PNG.')
            return
        extensao = '.webm' if self.tipo_atual == 'video' else f'.{self.formato_gif.get()}'
        descricao = ('WebM VP9 transparente', '*.webm') if extensao == '.webm' else (
            ('GIF animado transparente', '*.gif') if extensao == '.gif' else
            ('WebP animado com alfa suave', '*.webp'))
        destino = filedialog.asksaveasfilename(
            title='Exportar GIF/vídeo sem fundo', initialdir=str(self.caminho_atual.parent),
            initialfile=f'{self.caminho_atual.stem}_sem_fundo{extensao}',
            defaultextension=extensao, filetypes=[descricao])
        if not destino:
            return
        caminho_destino = Path(destino)
        if caminho_destino.resolve() == self.caminho_atual.resolve():
            messagebox.showerror('Destino inválido', 'Não sobrescreva o arquivo original.')
            return
        model = MODELOS[self.modelo.get()]
        self._marcar_ocupado(True, 'Preparando modelo local. Processando animação quadro a quadro...')
        threading.Thread(target=self._animacao_thread, daemon=True,
             args=(self.caminho_atual, caminho_destino, model,
                   self.descontaminar.get(), self._dados_ajustes(), self.formato_gif.get())).start()

    def _animacao_thread(self, entrada, destino, modelo, descontaminar, ajustes, formato_gif):
        import tempfile
        try:
            sessao = self._obter_sessao(modelo)
            destino.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.fundozero_', dir=destino.parent) as temporario:
                parcial = Path(temporario) / ('parcial' + destino.suffix)
                quadros = processar_midia(entrada, parcial, sessao, descontaminar, ajustes,
                    formato_gif=formato_gif,
                    avisar=lambda mensagem: self.mensagens.put(('status', mensagem)))
                os.replace(parcial, destino)
            self.mensagens.put(('midia_concluida', str(destino), quadros))
        except Exception:
            erro = traceback.format_exc()
            erro_no_log(erro)
            self.mensagens.put(('midia_erro', erro))

    def _processar_pasta(self):
        if self.ocupado:
            return
        entrada = filedialog.askdirectory(title='Selecione a pasta com imagens, GIFs e vídeos')
        if not entrada:
            return
        pasta_entrada = Path(entrada)
        arquivos = sorted((f for f in pasta_entrada.iterdir() if f.is_file() and
                           f.suffix.lower() in EXTENSOES_SUPORTADAS and not f.stem.endswith('_sem_fundo')),
                          key=lambda p: p.name.casefold())
        if not arquivos:
            messagebox.showinfo('Sem arquivos', 'A pasta não contém imagens, GIFs ou vídeos compatíveis.')
            return
        em_zip = self.modo_lote.get() == 'zip'
        if em_zip:
            escolha = filedialog.asksaveasfilename(
                title='Salvar imagens, GIFs e vídeos transparentes em ZIP',
                initialdir=entrada, initialfile='FundoZero_Resultados.zip',
                defaultextension='.zip', filetypes=[('Arquivo ZIP', '*.zip')])
        else:
            escolha = filedialog.askdirectory(title='Pasta onde salvar todos os resultados', initialdir=entrada)
        if not escolha:
            return
        destino = Path(escolha)
        if em_zip and destino.exists() and not messagebox.askyesno(
                'ZIP já existe', f'O arquivo {destino.name} já existe. Deseja substituí-lo?'):
            return
        tipo = 'um arquivo ZIP' if em_zip else 'arquivos separados'
        if not messagebox.askyesno('Remover fundos em lote',
            f'{len(arquivos)} arquivo(s) serão processado(s) com IA local.\n'
            f'Formato: {tipo}\nDestino: {destino}\n\nIniciar?'):
            return
        self._marcar_ocupado(True, f'Iniciando lote de {len(arquivos)} arquivos...')
        modelo = MODELOS[self.modelo.get()]
        args = (arquivos, destino, modelo, self.descontaminar.get(), self._dados_ajustes(), em_zip, self.formato_gif.get())
        threading.Thread(target=self._lote_thread, args=args, daemon=True).start()

    def _lote_thread(self, arquivos, destino, modelo, descontaminar, ajustes, em_zip, formato_gif):
        acertos, falhas = 0, []
        try:
            sessao = self._obter_sessao(modelo)
            acertos, falhas = gerar_lote(
                arquivos, destino, sessao, descontaminar, ajustes, em_zip,
                avisar=lambda status: self.mensagens.put(('lote_progress', status)),
                formato_gif=formato_gif)
        except Exception as exc:
            falhas.append(f'Falha ao preparar ou salvar o lote: {exc}')
            erro_no_log(traceback.format_exc())
        if falhas:
            erro_no_log('\n'.join(falhas))
        self.mensagens.put(('lote_concluido', acertos, falhas, str(destino), em_zip))

    def _fechar(self):
        if self.ocupado and not messagebox.askyesno('Processamento ativo',
                'A IA ainda está processando. Fechar agora interromperá a tarefa. Sair mesmo assim?'):
            return
        self.destroy()


if __name__ == '__main__':
    try:
        FundoZero().mainloop()
    except Exception:
        erro_no_log(traceback.format_exc())
        raise
