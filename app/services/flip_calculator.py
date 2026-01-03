"""
Flip Calculator Service
Implements all calculation logic from Bird Dog Flip Calculator
"""
from app.models.flip_calculator_models import (
    FlipCalculatorInput,
    FlipCalculatorResult,
    AcquisitionCosts,
    RenovationCosts,
    HoldingCosts,
    SellingCosts,
    ProfitAnalysis,
    REPAIR_COST_PER_SQFT,
    MINIMUM_ROI_THRESHOLD
)


class FlipCalculator:
    """
    Calculate flip deal profitability
    Based on Bird Dog Flip Calculator methodology
    """

    def __init__(self):
        self.repair_cost_per_sqft = REPAIR_COST_PER_SQFT

    def calculate(self, inputs: FlipCalculatorInput) -> FlipCalculatorResult:
        """
        Main calculation method
        Takes input parameters and returns complete analysis
        """

        # Step 1: Calculate Acquisition Costs
        acquisition = self._calculate_acquisition(inputs)

        # Step 2: Calculate Renovation Costs
        renovation = self._calculate_renovation(inputs)

        # Step 3: Calculate Holding Costs
        holding = self._calculate_holding(inputs, acquisition.total_acquisition, renovation.total_renovation)

        # Step 4: Calculate Selling Costs
        selling = self._calculate_selling(inputs)

        # Step 5: Calculate Profit Analysis
        profit_analysis = self._calculate_profit(
            inputs,
            acquisition.total_acquisition,
            renovation.total_renovation,
            holding.total_holding,
            selling.total_selling,
            holding.loan_amount,
            renovation.repair_cost
        )

        # Step 6: Calculate Recommendations
        max_offer_70 = self._calculate_max_offer_70_rule(inputs.arv, renovation.repair_cost)
        recommended_arv = self._calculate_recommended_arv(
            acquisition.total_acquisition,
            renovation.total_renovation,
            holding.total_holding,
            selling.total_selling
        )

        return FlipCalculatorResult(
            # Echo inputs
            property_address=inputs.property_address,
            city=inputs.city,
            zip_code=inputs.zip_code,
            sqft_living=inputs.sqft_living,
            purchase_price=inputs.purchase_price,
            arv=inputs.arv,
            hold_time_months=inputs.hold_time_months,
            # Calculated results
            acquisition=acquisition,
            renovation=renovation,
            holding=holding,
            selling=selling,
            profit_analysis=profit_analysis,
            # Recommendations
            max_offer_70_rule=max_offer_70,
            recommended_arv_for_profit=recommended_arv
        )

    def _calculate_acquisition(self, inputs: FlipCalculatorInput) -> AcquisitionCosts:
        """
        Calculate acquisition/buying costs
        Row 16-20 in Bird Dog calculator
        """
        purchase_price = inputs.purchase_price
        reinstatement_misc = inputs.reinstatement_misc
        closing_costs = purchase_price * inputs.closing_cost_rate_buying

        total_acquisition = purchase_price + reinstatement_misc + closing_costs

        return AcquisitionCosts(
            purchase_price=purchase_price,
            reinstatement_misc=reinstatement_misc,
            closing_costs=closing_costs,
            total_acquisition=total_acquisition
        )

    def _calculate_renovation(self, inputs: FlipCalculatorInput) -> RenovationCosts:
        """
        Calculate renovation costs
        Row 23-25 in Bird Dog calculator
        """
        # Repair cost: $45/sqft or override
        if inputs.repair_cost_override is not None:
            repair_cost = inputs.repair_cost_override
        else:
            repair_cost = inputs.sqft_living * self.repair_cost_per_sqft

        # Monthly maintenance over hold period
        monthly_maintenance_total = inputs.monthly_hoa_maintenance * inputs.hold_time_months

        total_renovation = repair_cost + monthly_maintenance_total

        return RenovationCosts(
            repair_cost=repair_cost,
            monthly_maintenance_total=monthly_maintenance_total,
            total_renovation=total_renovation
        )

    def _calculate_holding(
        self,
        inputs: FlipCalculatorInput,
        total_acquisition: float,
        total_renovation: float
    ) -> HoldingCosts:
        """
        Calculate holding costs
        Row 27-33 in Bird Dog calculator
        """
        # Loan amount (Loan-to-Cost)
        total_cost = total_acquisition + total_renovation
        loan_amount = total_cost * inputs.loan_to_cost_percentage

        # Interest payment (hard money/private money)
        # Formula: (loan_amount * annual_rate / 12) * months
        interest_payment = (loan_amount * inputs.interest_rate_annual / 12) * inputs.hold_time_months

        # Loan origination points (based on purchase price)
        loan_origination_points = inputs.purchase_price * inputs.loan_points

        # Property taxes (prorated)
        property_tax_prorated = (inputs.annual_property_tax / 12) * inputs.hold_time_months

        # Insurance (monthly total)
        insurance_total = inputs.monthly_insurance * inputs.hold_time_months

        # Utilities (monthly total)
        utilities_total = inputs.monthly_utilities * inputs.hold_time_months

        total_holding = (
            interest_payment +
            loan_origination_points +
            property_tax_prorated +
            insurance_total +
            utilities_total
        )

        return HoldingCosts(
            loan_amount=loan_amount,
            interest_payment=interest_payment,
            loan_origination_points=loan_origination_points,
            property_tax_prorated=property_tax_prorated,
            insurance_total=insurance_total,
            utilities_total=utilities_total,
            total_holding=total_holding
        )

    def _calculate_selling(self, inputs: FlipCalculatorInput) -> SellingCosts:
        """
        Calculate selling costs
        Row 36-41 in Bird Dog calculator
        All percentages based on ARV
        """
        arv = inputs.arv

        staging_marketing = inputs.staging_marketing
        closing_costs = arv * inputs.closing_cost_rate_selling
        seller_credit = arv * inputs.seller_credit_rate
        listing_commission = arv * inputs.listing_commission_rate
        buyer_commission = arv * inputs.buyer_commission_rate

        total_selling = (
            staging_marketing +
            closing_costs +
            seller_credit +
            listing_commission +
            buyer_commission
        )

        return SellingCosts(
            staging_marketing=staging_marketing,
            closing_costs=closing_costs,
            seller_credit=seller_credit,
            listing_commission=listing_commission,
            buyer_commission=buyer_commission,
            total_selling=total_selling
        )

    def _calculate_profit(
        self,
        inputs: FlipCalculatorInput,
        total_acquisition: float,
        total_renovation: float,
        total_holding: float,
        total_selling: float,
        loan_amount: float,
        repair_cost: float
    ) -> ProfitAnalysis:
        """
        Calculate profit and ROI
        Row 43-45 in Bird Dog calculator
        """
        # Total all costs
        total_all_costs = total_acquisition + total_renovation + total_holding + total_selling

        # Gross profit
        gross_profit = inputs.arv - total_all_costs

        # Profit margin %
        profit_margin_percent = (gross_profit / inputs.arv * 100) if inputs.arv > 0 else 0

        # ROI %
        roi_percent = (gross_profit / total_all_costs * 100) if total_all_costs > 0 else 0

        # Cash needed (if using financing)
        cash_needed = total_all_costs - loan_amount

        # Deal quality indicators
        is_profitable = gross_profit > 0
        meets_minimum_roi = roi_percent >= MINIMUM_ROI_THRESHOLD

        # 70% Rule: Purchase price should be <= (ARV * 0.70) - Repair Cost
        max_offer_70 = (inputs.arv * 0.70) - repair_cost
        meets_70_percent_rule = inputs.purchase_price <= max_offer_70

        return ProfitAnalysis(
            total_all_costs=total_all_costs,
            gross_profit=gross_profit,
            profit_margin_percent=round(profit_margin_percent, 2),
            roi_percent=round(roi_percent, 2),
            cash_needed=cash_needed,
            is_profitable=is_profitable,
            meets_minimum_roi=meets_minimum_roi,
            meets_70_percent_rule=meets_70_percent_rule
        )

    def _calculate_max_offer_70_rule(self, arv: float, repair_cost: float) -> float:
        """
        Calculate maximum offer using 70% rule
        Formula: (ARV × 0.70) - Repair Costs
        """
        return (arv * 0.70) - repair_cost

    def _calculate_recommended_arv(
        self,
        total_acquisition: float,
        total_renovation: float,
        total_holding: float,
        total_selling: float
    ) -> float:
        """
        Calculate recommended ARV for 15% profit margin
        Formula: Total Costs / 0.85
        """
        total_costs = total_acquisition + total_renovation + total_holding + total_selling
        # Assume 15% profit margin target
        return total_costs / 0.85


def calculate_flip_deal(inputs: FlipCalculatorInput) -> FlipCalculatorResult:
    """
    Convenience function to calculate flip deal
    """
    calculator = FlipCalculator()
    return calculator.calculate(inputs)


# Example usage
if __name__ == "__main__":
    # Example deal
    example_input = FlipCalculatorInput(
        property_address="7256 E Mckellips",
        city="Scottsdale",
        zip_code="85257",
        sqft_living=1800,
        purchase_price=450000,
        arv=570000,
        hold_time_months=5
    )

    result = calculate_flip_deal(example_input)

    print("="*80)
    print("FLIP DEAL ANALYSIS")
    print("="*80)
    print(f"\nProperty: {result.property_address}, {result.city}")
    print(f"Square Footage: {result.sqft_living:,} sqft")
    print(f"Purchase Price: ${result.purchase_price:,.2f}")
    print(f"ARV: ${result.arv:,.2f}")
    print(f"Hold Time: {result.hold_time_months} months")

    print(f"\n{'ACQUISITION COSTS':-^80}")
    print(f"Purchase Price:        ${result.acquisition.purchase_price:,.2f}")
    print(f"Closing Costs:         ${result.acquisition.closing_costs:,.2f}")
    print(f"Total Acquisition:     ${result.acquisition.total_acquisition:,.2f}")

    print(f"\n{'RENOVATION COSTS':-^80}")
    print(f"Repairs ({result.sqft_living} sqft × $45): ${result.renovation.repair_cost:,.2f}")
    print(f"Monthly Maintenance:   ${result.renovation.monthly_maintenance_total:,.2f}")
    print(f"Total Renovation:      ${result.renovation.total_renovation:,.2f}")

    print(f"\n{'HOLDING COSTS':-^80}")
    print(f"Loan Amount:           ${result.holding.loan_amount:,.2f}")
    print(f"Interest Payment:      ${result.holding.interest_payment:,.2f}")
    print(f"Loan Points:           ${result.holding.loan_origination_points:,.2f}")
    print(f"Property Tax:          ${result.holding.property_tax_prorated:,.2f}")
    print(f"Insurance:             ${result.holding.insurance_total:,.2f}")
    print(f"Utilities:             ${result.holding.utilities_total:,.2f}")
    print(f"Total Holding:         ${result.holding.total_holding:,.2f}")

    print(f"\n{'SELLING COSTS':-^80}")
    print(f"Staging/Marketing:     ${result.selling.staging_marketing:,.2f}")
    print(f"Closing Costs:         ${result.selling.closing_costs:,.2f}")
    print(f"Seller Credit:         ${result.selling.seller_credit:,.2f}")
    print(f"Listing Commission:    ${result.selling.listing_commission:,.2f}")
    print(f"Buyer Commission:      ${result.selling.buyer_commission:,.2f}")
    print(f"Total Selling:         ${result.selling.total_selling:,.2f}")

    print(f"\n{'PROFIT ANALYSIS':-^80}")
    print(f"Total All Costs:       ${result.profit_analysis.total_all_costs:,.2f}")
    print(f"Gross Profit:          ${result.profit_analysis.gross_profit:,.2f}")
    print(f"Profit Margin:         {result.profit_analysis.profit_margin_percent:.2f}%")
    print(f"ROI:                   {result.profit_analysis.roi_percent:.2f}%")
    print(f"Cash Needed:           ${result.profit_analysis.cash_needed:,.2f}")

    print(f"\n{'DEAL QUALITY':-^80}")
    print(f"Is Profitable:         {'✓ YES' if result.profit_analysis.is_profitable else '✗ NO'}")
    print(f"Meets 20% ROI:         {'✓ YES' if result.profit_analysis.meets_minimum_roi else '✗ NO'}")
    print(f"Meets 70% Rule:        {'✓ YES' if result.profit_analysis.meets_70_percent_rule else '✗ NO'}")

    print(f"\n{'RECOMMENDATIONS':-^80}")
    print(f"Max Offer (70% Rule):  ${result.max_offer_70_rule:,.2f}")
    print(f"Recommended ARV:       ${result.recommended_arv_for_profit:,.2f}")
    print("="*80)
