# Análise de Curva de Juros Brasileira (pq_rate_curve)

Este projeto em Julia realiza o download, processamento e análise de dados de títulos públicos prefixados brasileiros (LTN e NTN-F) para construir, interpolar e comparar curvas de rendimento (yield curves). Utiliza dados de duas fontes principais: Tesouro Direto (preços da manhã) e BACEN (preços médios).

## Funcionalidades

*   **Scraping de Dados:** Scripts para buscar dados de títulos do Tesouro Direto e via API do BACEN.
*   **Cálculos Financeiros:**
    *   Cálculo de fluxo de caixa para títulos com e sem cupom (NTN-F e LTN).
    *   Cálculo de taxas implícitas (yield-to-maturity).
*   **Construção de Curvas:**
    *   Construção de curva zero-cupom a partir de LTNs.
    *   Ajuste iterativo da curva usando NTN-Fs (bootstrapping).
    *   Geração de uma curva de rendimento completa e interpolada diariamente.
*   **Análise e Comparação:**
    *   Função para comparar taxas de duas curvas diferentes para prazos específicos.
    *   Geração de gráficos das curvas de juros.
    *   Exportação de resultados de comparação para arquivos Excel.

## Fontes de Dados

*   **Tesouro Direto:** Preços e taxas dos títulos disponíveis na manhã. (`scraping_tesouro_direto.jl`)
*   **BACEN:** Preços médios de negociação dos títulos públicos. (`scraping_tesouro_via_bacen.jl`)

## Estrutura do Projeto

*   `interest_rate_lib.jl`: Biblioteca principal com as funções de cálculo financeiro e construção/manipulação de curvas.
*   `interest_rate_main.jl`: Script principal que orquestra o processo de obtenção de dados, cálculo e geração de resultados.
*   `scraping_*.jl`: Scripts responsáveis pela coleta de dados das fontes.
*   `Project.toml` / `Manifest.toml`: Arquivos de ambiente Julia definindo as dependências.
*   `*.xlsx`: Arquivos de dados de entrada (se aplicável) e saída com resultados (ex: `comparison_curves_*.xlsx`).
*   `*.png`: Gráficos gerados das curvas de juros.

## Requisitos

*   Julia (versão recomendada especificada no `Project.toml` se houver)
*   Pacotes Julia listados no `Project.toml`, incluindo:
    *   `InterestRates`
    *   `BusinessDays`
    *   `Optim`
    *   `DataFrames`
    *   `HTTP` (provavelmente usado nos scrapers)
    *   `XLSX` (provavelmente usado para ler/escrever Excel)
    *   `Plots` (ou outra biblioteca de gráficos, se usada para os `.png`)
    *   `Logging`
    *   `Printf`
    *   `Dates`

## Como Usar

1.  **Instalar Dependências:**
    *   Navegue até a pasta do projeto no terminal.
    *   Inicie o Julia: `julia`
    *   Entre no modo de gerenciamento de pacotes pressionando `]`
    *   Ative o ambiente do projeto: `activate .`
    *   Instancie o ambiente (baixa e instala as dependências): `instantiate`
    *   Saia do modo de pacotes pressionando `Backspace`.

2.  **Executar Análise:**
    *   Execute o script principal: `include("interest_rate_main.jl")` dentro do REPL do Julia, ou `julia interest_rate_main.jl` no terminal.

3.  **Verificar Saídas:**
    *   Os resultados, como gráficos (`.png`) e tabelas de comparação (`.xlsx`), serão gerados na pasta do projeto.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
