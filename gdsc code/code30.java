import java.util.Scanner;

public class TicTacToe {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        
        char[][] board = new char[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                board[i][j] = ' ';
            }
        }

       
        printBoard(board);

        
        System.out.println("Enter your move (1-9): ");
        int playerMove = scanner.nextInt();

       
        board[playerMove / 3][playerMove % 3] = 'X';

       
        printBoard(board);

       
        if (hasWon(board, 'X')) {
            System.out.println("You win!");
            return;
        }

        
        int aiMove = getAIMove(board);

        
        board[aiMove / 3][aiMove % 3] = 'O';

        
        printBoard(board);

        
        if (hasWon(board, 'O')) {
            System.out.println("The AI wins!");
            return;
        }

       
        if (isDraw(board)) {
            System.out.println("The game is a draw!");
            return;
        }

        
        System.out.println("Enter your move (1-9): ");
        playerMove = scanner.nextInt();

        
        board[playerMove / 3][playerMove % 3] = 'X';

        
        printBoard(board);

        
        if (hasWon(board, 'X')) {
            System.out.println("You win!");
            return;
        }
