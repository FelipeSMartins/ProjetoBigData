# 📊 Explicação: Preço Normalizado Base 100

## 🎯 O que significa "Base 100"?

A normalização base 100 é uma técnica fundamental para análise comparativa de ativos financeiros. Esta metodologia transforma todos os preços para que o **primeiro dia do período analisado** tenha valor **100**.

### 📈 Interpretação dos valores:
- **100** = Preço inicial (dia de referência)
- **110** = Preço 10% maior que o inicial
- **90** = Preço 10% menor que o inicial
- **150** = Preço 50% maior que o inicial
- **75** = Preço 25% menor que o inicial

## 🔢 Como funciona o cálculo?

```
Preço Normalizado = (Preço Atual / Preço Inicial) × 100
```

### Exemplo prático:
Se o SPY começou em $400 e hoje está em $440:
```
Preço Normalizado = (440 / 400) × 100 = 110
```
Isso significa que o SPY teve uma valorização de **10%** no período.

## 🌟 Por que usar normalização?

### ✅ Vantagens principais:

#### 1. **Comparação Justa**
Permite comparar ativos com preços muito diferentes:
- SPY ($400) vs GLD ($180) vs VIX ($20)
- Todos começam em 100, facilitando a comparação visual

#### 2. **Visualização Clara**
Mostra performance relativa, não absoluta:
- Foca no **percentual de mudança**, não no valor em dólares
- Elimina a distorção causada por diferentes escalas de preço

#### 3. **Análise de Impacto**
Ideal para análise de eventos:
- Identifica claramente qual ativo foi mais/menos afetado
- Facilita a comparação de recuperação pós-evento

## 📊 Exemplo Comparativo

### Sem normalização (difícil de comparar):
| Ativo | Preço Inicial | Preço Final | Mudança |
|-------|---------------|-------------|---------|
| SPY   | $400          | $440        | +$40    |
| GLD   | $180          | $190        | +$10    |
| VIX   | $20           | $18         | -$2     |

### Com normalização base 100 (fácil de comparar):
| Ativo | Valor Inicial | Valor Final | Performance |
|-------|---------------|-------------|-------------|
| SPY   | 100           | 110         | +10%        |
| GLD   | 100           | 105.6       | +5.6%       |
| VIX   | 100           | 90          | -10%        |

## 🎯 Aplicação na Análise de Eventos

### Contexto dos eventos mundiais:
Quando analisamos o impacto de eventos como:
- **COVID-19 (Março 2020)**
- **Guerra na Ucrânia (Fevereiro 2022)**
- **Crise Financeira de 2008**

A normalização base 100 nos permite identificar:

#### ✅ **Durante o evento:**
- Qual ativo teve maior queda
- Qual se mostrou mais resiliente
- Qual teve maior volatilidade

#### ✅ **Após o evento:**
- Qual se recuperou mais rapidamente
- Qual teve melhor performance no longo prazo
- Qual manteve estabilidade

## 📈 Interpretação Prática

### Cenários típicos:

#### 🔴 **Impacto Negativo (Valor < 100)**
- **95**: Queda de 5%
- **85**: Queda de 15%
- **70**: Queda de 30%

#### 🟢 **Impacto Positivo (Valor > 100)**
- **105**: Alta de 5%
- **115**: Alta de 15%
- **130**: Alta de 30%

#### ⚪ **Estabilidade (Valor ≈ 100)**
- **98-102**: Variação mínima (±2%)

## 🔍 Vantagens na Análise de Portfolio

### 1. **Identificação de Padrões**
- Ativos que se movem juntos (correlação)
- Ativos que se movem em direções opostas (hedge)

### 2. **Análise de Risco**
- Volatilidade relativa entre ativos
- Comportamento em períodos de stress

### 3. **Tomada de Decisão**
- Comparação objetiva de performance
- Identificação de oportunidades de rebalanceamento

## 📚 Conclusão

A normalização base 100 é uma ferramenta essencial para:
- **Analistas financeiros** que precisam comparar ativos
- **Gestores de portfolio** que buscam otimização
- **Investidores** que querem entender performance relativa

Esta metodologia transforma dados complexos em informações visuais claras e comparáveis, facilitando a tomada de decisões informadas no mercado financeiro.

---

*Documento gerado automaticamente pelo sistema de análise de eventos financeiros*  
*Projeto Big Data - Análise de Impacto de Eventos Mundiais*