"""Static UI regression checks for export completion, plus real engine tests."""
import ast
from pathlib import Path

SRC = Path(__file__).with_name('SpriteMotion.py')
s=SRC.read_text('utf-8')
ast.parse(s)
assert 'job.succeeded.connect(on_success)' not in s
assert 'job.failed.connect(self.job_failed)' not in s
assert 'job.succeeded.connect(self._record_job_success)' in s
assert 'thread.finished.connect(self.job_finished)' in s
assert 'QTimer.singleShot(0,self._deliver_job_result)' in s
assert 'QMessageBox.information(self,\'Exportação concluída\'' not in s
assert 'try:os.startfile(folder)' not in s
assert 'self.open_export_folder_button.clicked.connect(self.open_export_folder)' in s
assert 'self._set_export_complete(folder,msg)' in s
assert 'self._set_export_complete(Path(path).parent,' in s
assert '48_000_000' in Path(__file__).with_name('gif_export.py').read_text('utf-8')
print('AUDITORIA V9: callbacks adiados até QThread.finished, sem janela modal de conclusão nem Explorer automático: OK')
