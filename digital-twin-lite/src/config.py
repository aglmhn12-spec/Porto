# src/config.py
from pydantic import BaseModel, Field


class SeparatorParams(BaseModel):
    # === Geometri separator ===
    A: float = Field(..., gt=0, description="Luas penampang (m^2)")
    V_total: float = Field(..., gt=0, description="Volume internal total separator (m^3)")
    L_max: float = Field(..., gt=0, description="Level maksimum yang dimodelkan (m)")
    V_min: float = Field(0.05, gt=0, description="Volume gas minimum untuk stabilitas numerik (m^3)")

    # === Properti cairan & lingkungan ===
    rho_l: float = Field(850.0, gt=0, description="Densitas cairan (kg/m^3)")
    g: float = Field(9.81, gt=0, description="Gravitasi (m/s^2)")
    Patm: float = Field(101_325.0, gt=0, description="Tekanan atmosfer (Pa)")
    P_out: float = Field(101_325.0, gt=0, description="Tekanan downstream outlet liquid (Pa)")

    # === Valve liquid outlet ===
    Cv_l: float = Field(..., ge=0, description="Koefisien valve liquid (m^3/s/sqrt(Pa))")

    # === Gas (ideal gas) ===
    R: float = Field(8.314, gt=0, description="Konstanta gas (J/mol/K)")
    T: float = Field(320.0, gt=0, description="Temperatur gas (K) - asumsi konstan (isothermal)")

    # === Gas outlet ===
    kg_n_sqrt: float = Field(..., ge=0, description="Koefisien outflow gas (mol/s/Pa)")
    n_out_max: float = Field(50.0, gt=0, description="Batas maksimum outflow gas (mol/s)")

    def gas_volume(self, L: float) -> float:
        """
        Volume gas = V_total - A*L, dibatasi minimal V_min
        """
        Vg = self.V_total - self.A * max(L, 0.0)
        return max(Vg, self.V_min)


class SimConfig(BaseModel):
    dt: float = Field(0.5, gt=0, description="Time step simulasi (s)")
    t_end: float = Field(600.0, gt=0, description="Durasi simulasi (s)")