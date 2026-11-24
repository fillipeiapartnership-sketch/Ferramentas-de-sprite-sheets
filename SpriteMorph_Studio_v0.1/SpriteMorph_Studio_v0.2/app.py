"""SpriteMorph Studio 0.2 GUI. Requires Python 3.10+, tkinter, Pillow, numpy."""
from __future__ import annotations

import os
from pathlib import Path
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image,ImageDraw,ImageFont,ImageTk

from engine import SpriteProject, Cancelled

BG='#0c1220';PANEL='#151f30';PANEL2='#1b293e';TEXT='#edf2ff';MUTED='#9cacbf';CYAN='#65d9ef';GOLD='#fbd184';BORDER='#2e3d54';DANGER='#ed848b'

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SpriteMorph Studio 0.2  |  Padronização de sprites 2D')
        self.geometry('1450x900');self.minsize(1110,730)
        self.configure(bg=BG)
        self.project=SpriteProject()
        self.messages=queue.Queue()
        self.worker=None
        self.busy=False
        self.current_group=None
        self.current_id=None
        self.last_selected=[]
        self.preview_request=0
        self.preview_busy=False
        self.preview_pending=False
        self.preview_cache={}
        self.animation_ids=[]
        self.animation_pos=0
        self.playing=False
        self.group_offset=0
        self.sprite_offset=0
        self.groups_display=[]
        self.style_widgets()
        self.build()
        self.after(100,self._poll)
        self.after(150,self._animate)

    def style_widgets(self):
        s=ttk.Style(self)
        s.theme_use('clam')
        s.configure('.',background=BG,foreground=TEXT,font=('Segoe UI',10))
        s.configure('TFrame',background=BG)
        s.configure('Panel.TFrame',background=PANEL)
        s.configure('TLabel',background=BG,foreground=TEXT)
        s.configure('Panel.TLabel',background=PANEL,foreground=TEXT)
        s.configure('Small.TLabel',background=PANEL,foreground=MUTED,font=('Segoe UI',9))
        s.configure('Title.TLabel',background=BG,foreground=TEXT,font=('Segoe UI Semibold',22))
        s.configure('TButton',background=PANEL2,foreground=TEXT,padding=(13,8),borderwidth=0,relief='flat')
        s.map('TButton',background=[('active','#294363'),('disabled','#1e2533')],foreground=[('disabled','#718098')])
        s.configure('Primary.TButton',background='#1c6682',foreground='white',padding=(15,9))
        s.map('Primary.TButton',background=[('active','#167a9c'),('disabled','#344c5b')])
        s.configure('TCheckbutton',background=PANEL,foreground=TEXT)
        s.map('TCheckbutton',background=[('active',PANEL)],foreground=[('active',TEXT)])
        s.configure('Treeview',background=PANEL,fieldbackground=PANEL,foreground=TEXT,rowheight=27,borderwidth=0,
                    font=('Segoe UI',9))
        s.configure('Treeview.Heading',background=PANEL2,foreground=CYAN,font=('Segoe UI Semibold',10),relief='flat')
        s.map('Treeview',background=[('selected','#245476')],foreground=[('selected','white')])
        s.configure('Horizontal.TProgressbar',troughcolor=PANEL2,background=CYAN)
        s.configure('TScale',background=PANEL,throughcolor=PANEL2)
        s.configure('TEntry',fieldbackground=PANEL2,foreground=TEXT,insertcolor=TEXT,padding=6)
        s.configure('TNotebook',background=BG,borderwidth=0)
        s.configure('TNotebook.Tab',background=PANEL,foreground=MUTED,padding=(16,9))
        s.map('TNotebook.Tab',background=[('selected',PANEL2)],foreground=[('selected',CYAN)])

    def build(self):
        head=ttk.Frame(self,padding=(20,16,20,8));head.pack(fill='x')
        ttk.Label(head,text='SpriteMorph Studio',style='Title.TLabel').pack(side='left')
        ttk.Label(head,text='  v0.2  •  processamento local  •  original preservado',foreground=MUTED).pack(side='left',padx=12)
        ttk.Button(head,text='Abrir ZIP do jogo',style='Primary.TButton',command=self.open_zip).pack(side='right')
        self.export_btn=ttk.Button(head,text='Exportar ZIP corrigido',command=self.export_zip,state='disabled')
        self.export_btn.pack(side='right',padx=10)
        controls=ttk.Frame(self,padding=(20,0,20,10));controls.pack(fill='x')
        self.stats=tk.StringVar(value='Abra um arquivo ZIP para iniciar a análise.')
        ttk.Label(controls,textvariable=self.stats,foreground=CYAN).pack(side='left')
        self.cancel_btn=ttk.Button(controls,text='Cancelar operação',command=self.cancel_operation,state='disabled')
        self.cancel_btn.pack(side='right')
        self.bar=ttk.Progressbar(self,orient='horizontal',mode='determinate',maximum=100)
        self.bar.pack(fill='x',padx=20)
        main=ttk.PanedWindow(self,orient='horizontal');main.pack(fill='both',expand=True,padx=20,pady=(14,8))
        left=ttk.Frame(main,style='Panel.TFrame',width=320);main.add(left,weight=1)
        mid=ttk.Frame(main,style='Panel.TFrame',width=400);main.add(mid,weight=1)
        right=ttk.Frame(main,style='Panel.TFrame',width=600);main.add(right,weight=2)
        self.make_groups(left)
        self.make_sprites(mid)
        self.make_preview(right)
        foot=ttk.Frame(self,padding=(20,0,20,15));foot.pack(fill='x')
        self.status=tk.StringVar(value='Pronto para importar seu projeto.')
        ttk.Label(foot,textvariable=self.status,foreground=MUTED).pack(side='left')
        ttk.Button(foot,text='Ver auditoria',command=self.show_audit).pack(side='right')

    def make_groups(self,parent):
        ttk.Label(parent,text='PERSONAGENS E GRUPOS',style='Panel.TLabel',font=('Segoe UI Semibold',12)).pack(anchor='w',padx=12,pady=(13,10))
        self.search_var=tk.StringVar()
        ent=ttk.Entry(parent,textvariable=self.search_var)
        ent.pack(fill='x',padx=12,pady=(0,8));self.search_var.trace_add('write',lambda *_:self.show_groups())
        self.group_tree=ttk.Treeview(parent,columns=('status','frames'),show='tree headings',selectmode='browse')
        self.group_tree.heading('#0',text='Grupo (filtrar acima)');self.group_tree.heading('status',text='Ativo');self.group_tree.heading('frames',text='Frames')
        self.group_tree.column('#0',width=205,stretch=True);self.group_tree.column('status',width=55,anchor='center',stretch=False)
        self.group_tree.column('frames',width=60,anchor='center',stretch=False)
        sc=ttk.Scrollbar(parent,orient='vertical',command=self.group_tree.yview)
        self.group_tree.configure(yscrollcommand=sc.set)
        sc.pack(side='right',fill='y');self.group_tree.pack(fill='both',expand=True,padx=(12,0),pady=2)
        self.group_tree.bind('<<TreeviewSelect>>',self.select_group)
        row=ttk.Frame(parent,style='Panel.TFrame');row.pack(fill='x',padx=12,pady=8)
        ttk.Button(row,text='Ativar / desativar',command=self.toggle_group).pack(side='left',expand=True,fill='x')
        ttk.Button(row,text='Revisar grupo',command=self.group_help).pack(side='left',padx=3)
        page=ttk.Frame(parent,style='Panel.TFrame');page.pack(fill='x',padx=12,pady=(0,12))
        ttk.Button(page,text='◀',command=lambda:self.move_group_page(-1)).pack(side='left')
        self.group_page_label=ttk.Label(page,text='0 / 0',style='Small.TLabel');self.group_page_label.pack(side='left',expand=True)
        ttk.Button(page,text='▶',command=lambda:self.move_group_page(1)).pack(side='right')

    def make_sprites(self,parent):
        ttk.Label(parent,text='SPRITES DO PERSONAGEM',style='Panel.TLabel',font=('Segoe UI Semibold',12)).pack(anchor='w',padx=12,pady=(13,8))
        self.group_title=tk.StringVar(value='Selecione um grupo à esquerda.')
        ttk.Label(parent,textvariable=self.group_title,style='Small.TLabel',wraplength=360).pack(anchor='w',padx=12,pady=(0,10))
        self.sprite_tree=ttk.Treeview(parent,columns=('animation','size','ref'),show='tree headings',selectmode='extended')
        self.sprite_tree.heading('#0',text='Arquivo / frame');self.sprite_tree.heading('animation',text='Ação')
        self.sprite_tree.heading('size',text='Conteúdo');self.sprite_tree.heading('ref',text='Ref.')
        self.sprite_tree.column('#0',width=190,stretch=True)
        self.sprite_tree.column('animation',width=90,stretch=False)
        self.sprite_tree.column('size',width=76,stretch=False)
        self.sprite_tree.column('ref',width=40,stretch=False,anchor='center')
        sc=ttk.Scrollbar(parent,orient='vertical',command=self.sprite_tree.yview);self.sprite_tree.configure(yscrollcommand=sc.set)
        sc.pack(side='right',fill='y');self.sprite_tree.pack(fill='both',expand=True,padx=(12,0))
        self.sprite_tree.bind('<<TreeviewSelect>>',self.select_sprite)
        row=ttk.Frame(parent,style='Panel.TFrame');row.pack(fill='x',padx=12,pady=(10,5))
        ttk.Button(row,text='Usar como referência',command=self.use_reference).pack(side='left',expand=True,fill='x')
        ttk.Button(row,text='Mover de grupo',command=self.move_selection).pack(side='left',padx=5)
        page=ttk.Frame(parent,style='Panel.TFrame');page.pack(fill='x',padx=12,pady=(0,13))
        ttk.Button(page,text='◀',command=lambda:self.move_sprite_page(-1)).pack(side='left')
        self.sprite_page_label=ttk.Label(page,text='0 / 0',style='Small.TLabel');self.sprite_page_label.pack(side='left',expand=True)
        ttk.Button(page,text='▶',command=lambda:self.move_sprite_page(1)).pack(side='right')

    def make_preview(self,parent):
        ttk.Label(parent,text='PRÉVIA  •  ANTES E DEPOIS',style='Panel.TLabel',font=('Segoe UI Semibold',12)).pack(anchor='w',padx=14,pady=(13,8))
        images=ttk.Frame(parent,style='Panel.TFrame');images.pack(fill='both',expand=True,padx=12,pady=4)
        self.before=tk.Canvas(images,bg='#0b1726',highlightthickness=1,highlightbackground=BORDER)
        self.before.pack(side='left',fill='both',expand=True,padx=(0,5))
        self.after_canvas=tk.Canvas(images,bg='#0b1726',highlightthickness=1,highlightbackground=BORDER)
        self.after_canvas.pack(side='left',fill='both',expand=True,padx=(5,0))
        self.preview_desc=tk.StringVar(value='A prévia aparecerá quando você selecionar um sprite.')
        ttk.Label(parent,textvariable=self.preview_desc,style='Small.TLabel',wraplength=540).pack(anchor='w',padx=12,pady=(10,7))
        self.play_btn=ttk.Button(parent,text='▶ Reproduzir animação',command=self.toggle_play)
        self.play_btn.pack(fill='x',padx=12,pady=(0,7))
        row=ttk.Frame(parent,style='Panel.TFrame');row.pack(fill='x',padx=12,pady=(0,6))
        ttk.Label(row,text='FPS',style='Small.TLabel').pack(side='left')
        self.fps=tk.IntVar(value=8)
        ttk.Scale(row,from_=2,to=24,orient='horizontal',variable=self.fps).pack(side='left',fill='x',expand=True,padx=(8,14))
        self.pixel_art=tk.BooleanVar(value=True)
        ttk.Checkbutton(row,text='Pixel art (sem suavização)',variable=self.pixel_art,command=self.update_preview).pack(side='left')
        row2=ttk.Frame(parent,style='Panel.TFrame');row2.pack(fill='x',padx=12,pady=(0,5))
        ttk.Label(row2,text='Intensidade da correção',style='Small.TLabel').pack(side='left')
        self.strength=tk.DoubleVar(value=1.0)
        ttk.Scale(row2,from_=0.,to=1.,orient='horizontal',variable=self.strength,command=lambda _ : self.update_preview()).pack(side='left',fill='x',expand=True,padx=10)
        opts=ttk.Frame(parent,style='Panel.TFrame');opts.pack(fill='x',padx=12,pady=(5,4))
        ttk.Label(opts,text='Altura-alvo do personagem (contorno, px)',style='Small.TLabel').pack(side='left')
        self.target_px=tk.StringVar(value='0')
        target_entry=ttk.Entry(opts,textvariable=self.target_px,width=9)
        target_entry.pack(side='left',padx=(10,4))
        target_entry.bind('<Return>',lambda _ :self.update_preview())
        target_entry.bind('<FocusOut>',lambda _ :self.update_preview())
        self.expand_var=tk.BooleanVar(value=False)
        ttk.Checkbutton(parent,text='Permitir ampliar canvas dos quadros (somente atlas de grade confirmada)',
                        variable=self.expand_var,command=self.update_preview).pack(anchor='w',padx=13,pady=(4,5))
        self.adapt_var=tk.BooleanVar(value=True)
        ttk.Checkbutton(parent,text='Adaptar o motor de escala do Metro Fighters quando detectado',
                        variable=self.adapt_var,command=self.update_preview).pack(anchor='w',padx=13,pady=(0,5))
        ttk.Label(parent,text='Escala uniforme e contorno real. Poses agachadas ou deitadas mantêm a anatomia. '
                  'Ampliar o canvas requer que o motor do jogo aceite dimensões variáveis.',
                  style='Small.TLabel',wraplength=540).pack(anchor='w',padx=12,pady=(0,15))
        self.before.bind('<Configure>',lambda e:self.draw_preview())
        self.after_canvas.bind('<Configure>',lambda e:self.draw_preview())
        self.before_img=None;self.after_img=None
        self.draw_preview()

    def _setbusy(self,value:bool):
        self.busy=value
        self.cancel_btn.configure(state=('normal' if value else 'disabled'))
        self.export_btn.configure(state=('normal' if not value and self.project.input_path else 'disabled'))

    def _start(self,method,done):
        if self.busy:return
        self._setbusy(True)
        self.bar['value']=0
        def runner():
            try:
                result=method()
                self.messages.put(('done',done,result))
            except Cancelled as ex:
                self.messages.put(('cancel',str(ex)))
            except Exception:
                self.messages.put(('error',traceback.format_exc()))
        self.worker=threading.Thread(target=runner,daemon=True)
        self.worker.start()

    def _notify(self,label,current,total):
        self.messages.put(('progress',label,current,total))

    def _poll(self):
        try:
            while True:
                message=self.messages.get_nowait()
                if message[0]=='progress':
                    _,label,a,b=message
                    self.bar['value']=a/max(b,1)*100
                    self.status.set(f'{label}: {a:,} / {b:,}')
                elif message[0]=='done':
                    _,done,result=message
                    self._setbusy(False);done(result)
                elif message[0]=='cancel':
                    self._setbusy(False);self.status.set(message[1]);self.bar['value']=0
                elif message[0]=='error':
                    self._setbusy(False);self.status.set('Ocorreu um erro. Veja detalhes na janela.')
                    messagebox.showerror('Erro na operação',message[1][-5000:])
                elif message[0]=='preview':
                    _,request,it_id,orig,mod,audit=message
                    self.preview_busy=False
                    if request==self.preview_request and it_id==self.current_id:
                        self.before_img,self.after_img=orig,mod
                        self.draw_preview()
                        s=self.project.items[it_id]
                        factor=audit['scale_applied']
                        text=f'Arquivo: {Path(s.file).name}  |  frame {s.frame+1}  |  escala aplicada: {factor:.3f}x'
                        if audit['warnings']:text+='  |  '+', '.join(audit['warnings'])
                        self.preview_desc.set(text)
                    if self.preview_pending:
                        self.preview_pending=False
                        self.update_preview()
                elif message[0]=='preview_err':
                    _,request,error=message
                    self.preview_busy=False
                    if request==self.preview_request:self.preview_desc.set(error)
                    if self.preview_pending:
                        self.preview_pending=False
                        self.update_preview()
        except queue.Empty:
            pass
        self.after(100,self._poll)

    def open_zip(self):
        if self.busy:return
        p=filedialog.askopenfilename(title='Escolha o ZIP do jogo 2D',filetypes=[('Arquivos ZIP','*.zip')])
        if not p:return
        self.playing=False;self.preview_request+=1
        self.project=SpriteProject();self.preview_cache={};self.current_id=None;self.current_group=None;self.preview_pending=False
        self.group_tree.delete(*self.group_tree.get_children())
        self.sprite_tree.delete(*self.sprite_tree.get_children())
        self.stats.set('Analisando ZIP: '+Path(p).name)
        self._start(lambda:self.project.scan(p,self._notify),self.scan_done)

    def scan_done(self,stats):
        self.bar['value']=100
        self.stats.set(f"{stats['imagens_encontradas']:,} imagens  |  "
                       f"{stats.get('sprites_analisados',0):,} quadros  |  "
                       f"{stats.get('grupos_ativos',0)} personagens  |  "
                       f"{stats.get('atlas_com_metadados',0)} atlas declarados  |  "
                       f"motor: {stats.get('motor_do_jogo_reconhecido','generico')}")
        self.target_px.set(str(int(round(self.project.target_height))))
        self.expand_var.set(self.project.game_adapter=='metro_fighters')
        self.adapt_var.set(self.project.game_adapter=='metro_fighters')
        self.project.canvas_expand=bool(self.expand_var.get())
        self.project.adapt_engine=bool(self.adapt_var.get())
        self.show_groups()
        self.status.set(f"Análise concluída em {stats['segundos_analise']} s. Confira os grupos e as referências antes de exportar.")
        if self.project.groups:
            first=next(iter(self.group_tree.get_children()),None)
            if first:self.group_tree.selection_set(first);self.select_group()
        else:
            messagebox.showwarning('Nenhum candidato','Não foram detectados sprites elegíveis com transparência. '
                                   'O ZIP pode ter atlas sem separação, cenários ou formatos proprietários.')

    def show_groups(self):
        groups=sorted(self.project.groups.values(),key=lambda g:(not g.enabled,g.key.casefold()))
        filter_=self.search_var.get().casefold()
        self.groups_display=[g for g in groups if filter_ in g.key.casefold()]
        self.group_offset=min(self.group_offset,max(0,(len(self.groups_display)-1)//250*250))
        self.group_tree.delete(*self.group_tree.get_children())
        for g in self.groups_display[self.group_offset:self.group_offset+250]:
            label=g.key[-65:] if len(g.key)>65 else g.key
            self.group_tree.insert('', 'end',iid=g.key,text=label,values=('✓' if g.enabled else '○',len(g.ids)))
        self.group_page_label.configure(text=f'{self.group_offset+1 if self.groups_display else 0}–{min(len(self.groups_display),self.group_offset+250)} / {len(self.groups_display)}')
        if self.current_group in self.group_tree.get_children():self.group_tree.selection_set(self.current_group)

    def move_group_page(self,step):
        self.group_offset=max(0,min((len(self.groups_display)-1)//250*250,self.group_offset+step*250)) if self.groups_display else 0
        self.show_groups()

    def select_group(self,_=None):
        selected=self.group_tree.selection()
        if not selected:return
        key=selected[0]
        if key not in self.project.groups:return
        self.current_group=key
        g=self.project.groups[key]
        self.group_title.set(f'{g.key}  •  confiança {g.confidence}  •  {len(g.ids)} quadros  •  referência: {self.project.group_baselines.get(g.key,0):.0f}px')
        self.sprite_offset=0
        self.show_sprites()
        if g.ids:
            first=next(iter(self.sprite_tree.get_children()),None)
            if first:
                children=self.sprite_tree.get_children(first)
                if children:self.sprite_tree.selection_set(children[0]);self.select_sprite()

    def show_sprites(self):
        self.sprite_tree.delete(*self.sprite_tree.get_children())
        if not self.current_group or self.current_group not in self.project.groups:return
        g=self.project.groups[self.current_group]
        ids=g.ids
        self.sprite_offset=min(self.sprite_offset,max(0,(len(ids)-1)//250*250))
        bymotion={}
        for sid in ids[self.sprite_offset:self.sprite_offset+250]:
            motion=self.project.items[sid].category
            bymotion.setdefault(motion,[]).append(sid)
        priority=['idle','walk','run','block','punch','kick','upper','hit','jump','dive','fall','getup','stun','death','sem_acao']
        for motion in sorted(bymotion,key=lambda v:(priority.index(v) if v in priority else 30,v)):
            parent_id='__motion__'+motion
            self.sprite_tree.insert('','end',iid=parent_id,text=f'▾ {motion.upper()} ({len(bymotion[motion])})',
                                    values=('MOVIMENTO','', ''),open=True)
            for sid in bymotion[motion]:
                it=self.project.items[sid]
                short=Path(it.file).name[:34]+(f' • {it.frame+1}' if it.frame else '')
                if it.cell:short+=f' • quadro {ids.index(sid)+1}'
                self.sprite_tree.insert(parent_id,'end',iid=sid,text=short,values=(it.category[:15],
                                      f'{it.width_of_subject}×{it.height_of_subject}',
                                      '★' if g.reference==sid else ''))
        self.sprite_page_label.configure(text=f'{self.sprite_offset+1 if ids else 0}–{min(len(ids),self.sprite_offset+250)} / {len(ids)}')

    def move_sprite_page(self,step):
        if not self.current_group:return
        n=len(self.project.groups[self.current_group].ids)
        self.sprite_offset=max(0,min((n-1)//250*250,self.sprite_offset+step*250)) if n else 0
        self.show_sprites()

    def select_sprite(self,_=None):
        sel=self.sprite_tree.selection()
        sel=[sid for sid in sel if sid in self.project.items]
        if not sel:return
        self.last_selected=list(sel)
        self.current_id=sel[0]
        it=self.project.items[self.current_id]
        g=self.project.groups.get(it.group)
        self.animation_ids=[sid for sid in (g.ids if g else [it.id]) if self.project.items[sid].track==it.track]
        if len(self.animation_ids)==1 and it.cell:
            self.animation_ids=[sid for sid in g.ids if self.project.items[sid].file==it.file]
        self.animation_pos=self.animation_ids.index(it.id) if it.id in self.animation_ids else 0
        self.update_preview()

    def update_preview(self):
        if not self.current_id or self.busy:return
        try:self._apply_scale_settings()
        except (ValueError,tk.TclError) as exc:
            self.preview_desc.set(str(exc));return
        if self.preview_busy:
            self.preview_pending=True
            return
        self.preview_request+=1;req=self.preview_request
        sid=self.current_id;strength=float(self.strength.get());pix=bool(self.pixel_art.get())
        self.preview_busy=True
        def runner():
            try:
                a,b,au=self.project.preview(sid,strength,pix)
                self.messages.put(('preview',req,sid,a,b,au))
            except Exception as exc:self.messages.put(('preview_err',req,str(exc)))
        threading.Thread(target=runner,daemon=True).start()

    def _apply_scale_settings(self):
        t=float(self.target_px.get().replace(',','.'))
        self.project.set_target_height(t)
        self.project.canvas_expand=bool(self.expand_var.get())
        self.project.adapt_engine=bool(self.adapt_var.get())

    def _draw_image(self,canvas:tk.Canvas,img:Image.Image|None,label:str):
        canvas.delete('all')
        w=max(40,canvas.winfo_width());h=max(40,canvas.winfo_height())
        # Subtle linear gradient for viewport, without checkerboard textures.
        bg=Image.new('RGB',(w,h))
        d=ImageDraw.Draw(bg)
        for y in range(h):
            t=y/max(1,h-1)
            c=(int(11+10*t),int(21+13*t),int(36+17*t))
            d.line((0,y,w,y),fill=c)
        d.text((10,10),label,fill='#a8dff4')
        if img is not None:
            pad=16;space_w=max(1,w-2*pad);space_h=max(1,h-60)
            scale=min(space_w/max(1,img.width),space_h/max(1,img.height))
            if self.pixel_art.get() and scale>=1:
                scale=max(1,int(scale))
            size=(max(1,round(img.width*scale)),max(1,round(img.height*scale)))
            # Guard huge photo allocations when user resizes window
            thumb=img.resize(size,Image.Resampling.NEAREST if self.pixel_art.get() else Image.Resampling.LANCZOS)
            x=(w-size[0])//2;y=(h-size[1])//2+10
            bg.paste(thumb,(x,y),thumb)
            d=ImageDraw.Draw(bg)
            d.text((10,h-23),f'{img.width} × {img.height} px',fill='#b9c9de')
            thumb.close()
        photo=ImageTk.PhotoImage(bg)
        canvas.image=photo
        canvas.create_image(0,0,anchor='nw',image=photo)
        bg.close()

    def draw_preview(self):
        if not hasattr(self,'before'):return
        self._draw_image(self.before,self.before_img,'ORIGINAL')
        self._draw_image(self.after_canvas,self.after_img,'NORMALIZADO')

    def toggle_play(self):
        if not self.animation_ids:return
        self.playing=not self.playing
        self.play_btn.configure(text='⏸ Parar animação' if self.playing else '▶ Reproduzir animação')

    def _animate(self):
        if self.playing and len(self.animation_ids)>1 and not self.busy and not self.preview_busy:
            self.animation_pos=(self.animation_pos+1)%len(self.animation_ids)
            self.current_id=self.animation_ids[self.animation_pos]
            self.update_preview()
        self.after(max(40,int(1000/max(1,self.fps.get()))),self._animate)

    def toggle_group(self):
        if not self.current_group:return
        g=self.project.groups[self.current_group]
        g.enabled=not g.enabled
        self.project._refresh_calibration()
        self.show_groups();self.update_preview()
        self.status.set(f'Grupo {g.key}: '+('ativado' if g.enabled else 'desativado'))

    def use_reference(self):
        if not self.current_id or not self.current_group:return
        self.project.set_reference(self.current_group,self.current_id)
        self.show_sprites();self.update_preview()
        self.status.set('Nova referência definida: '+self.project.items[self.current_id].file)

    def move_selection(self):
        if not self.last_selected or not self.current_group:return
        target=simpledialog.askstring('Agrupar manualmente','Nome do personagem de destino (existente ou novo):',
                                      initialvalue=self.current_group,parent=self)
        if not target:return
        try:
            selected=list(self.last_selected)
            self.project.assign_to_group(selected,target)
            self.current_group=target
            self.project._refresh_calibration()
            self.show_groups();self.show_sprites();self.update_preview()
            self.status.set(f'{len(selected)} sprite(s) transferido(s) para {target}.')
        except Exception as exc:messagebox.showerror('Não foi possível mover',str(exc))

    def group_help(self):
        messagebox.showinfo('Revisar agrupamentos','Cada linha à esquerda é um personagem estimado. '
                            'Selecione um grupo e confira os sprites no centro.\n\n'
                            'Selecione um quadro em pé e clique em “Usar como referência”.\n\n'
                            'Selecione vários quadros (Ctrl ou Shift) e mova para outro grupo, se necessário.\n\n'
                            'Somente grupos ativos serão corrigidos. Grupos incertos e imagens não identificadas ficam desativados por segurança.\n\n'
                            'As animações são dimensionadas por sequência, sem distorcer as poses.')

    def export_zip(self):
        if self.busy or not self.project.input_path:return
        try:self._apply_scale_settings()
        except (ValueError,tk.TclError) as exc:
            messagebox.showerror('Altura-alvo inválida',str(exc));return
        active=sum(g.enabled for g in self.project.groups.values())
        if not active:
            messagebox.showwarning('Nada para corrigir','Ative ao menos um grupo com animações antes de exportar.')
            return
        inp=Path(self.project.input_path)
        p=filedialog.asksaveasfilename(title='Salvar ZIP corrigido',defaultextension='.zip',
                                        initialfile=inp.stem+'_normalizado.zip',filetypes=[('Arquivos ZIP','*.zip')])
        if not p:return
        if Path(p).resolve()==inp.resolve():
            messagebox.showerror('Arquivo original protegido','Escolha outro nome. O ZIP original nunca será sobrescrito.')
            return
        if not messagebox.askyesno('Exportar correções',f'Aplicar correções aos {active} grupo(s) ativo(s)?\n\n'
                                 'Os arquivos de cenário, áudio e código do jogo serão preservados.\n'
                                 'Um relatório de auditoria JSON será salvo junto do novo ZIP.'):
            return
        self.playing=False
        self._start(lambda:self.project.export(p,float(self.strength.get()),bool(self.pixel_art.get()),self._notify,True),self.export_done)

    def export_done(self,report):
        self.bar['value']=100
        self.status.set(f"Exportação concluída: {report['arquivos_editados']} arquivos e "
                        f"{report['frames_ajustados']} frames corrigidos. Auditoria CRC: OK.")
        msg=(f"ZIP corrigido salvo em:\n{report['arquivo_exportado']}\n\n"
             f"Arquivos alterados: {report['arquivos_editados']} (inclui o JS adaptado, quando aplicável)\n"
             f"Quadros ajustados: {report['frames_ajustados']}\n"
             f"Alertas: {len(report['alertas_de_auditoria'])}\n"
             f"Falhas de processamento: {len(report['falhas_de_processamento'])}\n\n"
             'Veja também o arquivo de auditoria JSON na mesma pasta.')
        messagebox.showinfo('Exportação e auditoria concluídas',msg)

    def show_audit(self):
        report=self.project.audit_report
        if not report:
            messagebox.showinfo('Auditoria','A auditoria completa acontece após a exportação.\n\n'
                                f"Avisos da análise atual: {len(self.project.warnings)}")
            return
        top=tk.Toplevel(self);top.title('Relatório de auditoria');top.geometry('750x530');top.configure(bg=BG)
        txt=tk.Text(top,background=PANEL,foreground=TEXT,insertbackground=CYAN,font=('Consolas',10),wrap='word')
        txt.pack(fill='both',expand=True,padx=15,pady=15)
        import json
        txt.insert('1.0',json.dumps(report,ensure_ascii=False,indent=2))
        txt.configure(state='disabled')

    def cancel_operation(self):
        if self.busy:
            self.project.cancel.set()
            self.status.set('Solicitação de cancelamento enviada. Finalizando o arquivo atual...')

if __name__=='__main__':
    try:
        from PIL import ImageTk
        app=App();app.mainloop()
    except Exception:
        traceback.print_exc()
        input('Erro ao abrir a interface. Pressione Enter para sair...')
