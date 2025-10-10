# Análise de Código Obsoleto

## 📋 Scripts de Análise/Diagnóstico (Criar pasta `analysis/`)

Estes scripts foram criados para análise pontual e devem ser movidos para `analysis/`:

```bash
mkdir -p analysis
mv analyze_high_cost_days.jl analysis/
mv analyze_min_bonds_impact.jl analysis/
mv compare_problematic_day.jl analysis/
mv check_db.jl analysis/
```

**Razão**: Scripts úteis para debug mas não são parte do workflow principal.

## 🧪 Scripts de Teste (Criar pasta `tests/` ou remover)

Scripts criados apenas para testar funcionalidades durante desenvolvimento:

```bash
# Opção 1: Mover para tests/
mv test_fixed_persistence.jl tests/
mv test_get_rate_final.jl tests/
mv test_min_bonds_fix.jl tests/

# Opção 2: Remover (já validados)
rm test_fixed_persistence.jl test_get_rate_final.jl test_min_bonds_fix.jl
```

**Recomendação**: **Remover** (já validamos que funcionam e não são testes automatizados).

## 🎬 Scripts de Animação Duplicados

Temos 3 scripts de animação com funções sobrepostas:

1. **`create_yield_curve_animation.jl`** (original) - lê CSV
2. **`create_animation_from_db.jl`** (novo) - lê banco, versão rápida
3. **`create_animation_from_db_full.jl`** (novo) - lê banco, 1 frame/curva

**Recomendação**: Manter todos 3, mas documentar claramente:
- Original: para CSVs específicos
- `_db.jl`: animação rápida do banco completo
- `_db_full.jl`: animação completa sem pulos

## ✅ Scripts de Produção (Manter)

Estes são essenciais e devem permanecer no root:

- ✅ **`build_historical_curves.jl`** - Constrói banco histórico
- ✅ **`fit_curvas.jl`** - Fit de curvas CLI
- ✅ **`run_continuous_walkforward_cv.jl`** - Validação de hiperparâmetros
- ✅ **`create_yield_curve_animation.jl`** - Animação CLI (original)
- ✅ **`create_animation_from_db.jl`** - Animação do banco (rápida)
- ✅ **`create_animation_from_db_full.jl`** - Animação completa do banco

## 📝 Resumo de Ações Recomendadas

```bash
# 1. Criar pasta de análise
mkdir -p analysis

# 2. Mover scripts de análise
mv analyze_high_cost_days.jl analysis/
mv analyze_min_bonds_impact.jl analysis/
mv compare_problematic_day.jl analysis/
mv check_db.jl analysis/

# 3. Remover testes pontuais (já validados)
rm test_fixed_persistence.jl
rm test_get_rate_final.jl
rm test_min_bonds_fix.jl

# 4. Manter scripts de produção no root (nenhuma ação)
```

## 🔍 Código Obsoleto no Módulo Principal

### `src/persistence.jl`

✅ **Nenhum código obsoleto detectado** - código recém-refatorado e funcionando.

### `src/high_level_api.jl`

✅ **Nenhum código obsoleto detectado** - recém-corrigido com verificação pós-outliers.

### `config.toml`

✅ **Atualizado** - `min_bonds_for_fit = 6` já configurado.

## 📊 Estatísticas

- **Scripts de produção**: 6 (manter)
- **Scripts de análise**: 4 (mover para `analysis/`)
- **Scripts de teste**: 3 (remover)
- **Código obsoleto no módulo**: 0 ✅
