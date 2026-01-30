// SOLFUNMEME Token Airdrop Program
use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};

declare_id!("SOLFUN22826592eb0802f591fba3c406ec4c23a807b39f");

#[program]
pub mod solfunmeme_airdrop {
    use super::*;
    
    pub fn initialize(ctx: Context<Initialize>, total_supply: u64) -> Result<()> {
        let airdrop = &mut ctx.accounts.airdrop;
        airdrop.authority = ctx.accounts.authority.key();
        airdrop.total_supply = total_supply;
        airdrop.distributed = 0;
        Ok(())
    }
    
    pub fn claim_airdrop(ctx: Context<ClaimAirdrop>, amount: u64) -> Result<()> {
        let airdrop = &mut ctx.accounts.airdrop;
        
        require!(
            airdrop.distributed + amount <= airdrop.total_supply,
            ErrorCode::InsufficientSupply
        );
        
        // Transfer tokens
        let cpi_accounts = Transfer {
            from: ctx.accounts.vault.to_account_info(),
            to: ctx.accounts.recipient.to_account_info(),
            authority: ctx.accounts.authority.to_account_info(),
        };
        
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        
        token::transfer(cpi_ctx, amount)?;
        
        airdrop.distributed += amount;
        
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(init, payer = authority, space = 8 + 32 + 8 + 8)]
    pub airdrop: Account<'info, AirdropState>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ClaimAirdrop<'info> {
    #[account(mut)]
    pub airdrop: Account<'info, AirdropState>,
    #[account(mut)]
    pub vault: Account<'info, TokenAccount>,
    #[account(mut)]
    pub recipient: Account<'info, TokenAccount>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[account]
pub struct AirdropState {
    pub authority: Pubkey,
    pub total_supply: u64,
    pub distributed: u64,
}

#[error_code]
pub enum ErrorCode {
    #[msg("Insufficient supply for airdrop")]
    InsufficientSupply,
}

// Airdrop recipients (1 authors)
// Total: 36,744 SOLFUNMEME tokens
